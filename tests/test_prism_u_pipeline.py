"""Unit tests for Phase 12 (Acquisition/PRISM-U policy), Phase 14 (Calibration), and Phase 15 (Sequential Stopping)."""
from __future__ import annotations

import numpy as np
import pytest

from audit.acquisition import AdditiveAcquisitionScorer, RatioAcquisitionScorer
from audit.evidence import SurrogateEvidenceModel
from audit.prism_u import (
    AdaptiveGainStaticPrivacyPolicy,
    NoFeasibleQueriesError,
    PRISMUPolicy,
    StaticPrivacyAwarePolicy,
)
from audit.stopping import AuditResult, SequentialStoppingEngine, StoppingConfig
from evaluation.calibration import CalibratedThresholds, TemperatureCalibrator, ThresholdCalibrator
from models.black_box_target import CachedBlackBoxTarget


def _make_synthetic_surrogate(seed: int = 42):
    """Create synthetic surrogate bank with 12 HC, 48 HN over 30 queries, 5 classes."""
    rng = np.random.default_rng(seed)
    n_hc, n_hn, n_q, n_c = 12, 48, 30, 5
    hc = rng.dirichlet(np.ones(n_c) * 4, size=(n_hc, n_q)).astype(np.float64)
    hn = rng.dirichlet(np.ones(n_c) * 4, size=(n_hn, n_q)).astype(np.float64)

    # Make query 0 and 1 very informative for HC vs HN
    hc[:, 0, :] = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
    hn[:, 0, :] = np.array([0.05, 0.8, 0.05, 0.05, 0.05])

    hc[:, 1, :] = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
    hn[:, 1, :] = np.array([0.1, 0.7, 0.1, 0.05, 0.05])

    # Add small jitter and normalize
    hc = np.clip(hc + rng.normal(0, 0.01, size=hc.shape), 1e-4, 1.0)
    hn = np.clip(hn + rng.normal(0, 0.01, size=hn.shape), 1e-4, 1.0)
    hc /= hc.sum(axis=-1, keepdims=True)
    hn /= hn.sum(axis=-1, keepdims=True)

    return SurrogateEvidenceModel(hc_probs=hc, hn_probs=hn, bandwidth=1.0)


# ---------------------------------------------------------------------------
# Test Phase 12: Acquisition & Policies
# ---------------------------------------------------------------------------

class TestAcquisitionScorers:
    def test_ratio_scorer(self):
        scorer = RatioAcquisitionScorer(eps=1e-4)
        gains = np.array([0.5, 0.5])
        risks = np.array([0.01, 0.20])
        scores = scorer.score(gains, risks)
        assert scores[0] > scores[1]  # Lower risk -> higher ratio score

    def test_additive_scorer(self):
        scorer = AdditiveAcquisitionScorer(penalty_lambda=2.0)
        gains = np.array([0.8, 0.5])
        risks = np.array([0.1, 0.1])
        scores = scorer.score(gains, risks)
        assert scores[0] > scores[1]

    def test_additive_scorer_adaptive_penalty(self):
        scorer = AdditiveAcquisitionScorer(penalty_lambda=1.0, adaptive_penalty=True)
        gains = np.array([0.5])
        risks = np.array([0.1])
        # With small remaining budget, penalty should be higher
        score_tight = scorer.score(gains, risks, remaining_privacy_budget=0.05)
        score_loose = scorer.score(gains, risks, remaining_privacy_budget=0.50)
        assert score_loose[0] > score_tight[0]


class TestPRISMUPolicies:
    def test_static_privacy_aware_policy(self):
        ev = _make_synthetic_surrogate()
        pol = StaticPrivacyAwarePolicy(pool_size=30, evidence_model=ev, penalty_lambda=1.0)
        t = CachedBlackBoxTarget(
            prob_matrix=ev.hc_probs.mean(axis=0),
            target_id="target_0",
            query_budget=5,
            query_indices=list(range(30)),
        )
        q1 = pol.select_next(t.transcript)
        assert 0 <= q1 < 30
        t.query(q1)
        q2 = pol.select_next(t.transcript)
        assert q2 != q1

    def test_adaptive_gain_static_privacy_policy(self):
        ev = _make_synthetic_surrogate()
        pol = AdaptiveGainStaticPrivacyPolicy(pool_size=30, evidence_model=ev, penalty_lambda=1.0)
        t = CachedBlackBoxTarget(
            prob_matrix=ev.hc_probs.mean(axis=0),
            target_id="target_0",
            query_budget=5,
            query_indices=list(range(30)),
        )
        q1 = pol.select_next(t.transcript)
        assert 0 <= q1 < 30
        t.query(q1)
        q2 = pol.select_next(t.transcript)
        assert q2 != q1

    def test_full_prism_u_policy_selection(self):
        ev = _make_synthetic_surrogate()
        pol = PRISMUPolicy(pool_size=30, evidence_model=ev, privacy_budget=float("inf"))
        t = CachedBlackBoxTarget(
            prob_matrix=ev.hc_probs.mean(axis=0),
            target_id="target_0",
            query_budget=5,
            query_indices=list(range(30)),
        )
        q1 = pol.select_next(t.transcript)
        assert 0 <= q1 < 30
        t.query(q1)
        q2 = pol.select_next(t.transcript)
        assert q2 != q1

    def test_prism_u_hard_privacy_budget_feasibility(self):
        ev = _make_synthetic_surrogate()
        # Set an extremely restrictive privacy budget (0.50)
        pol = PRISMUPolicy(pool_size=30, evidence_model=ev, privacy_budget=0.50)
        t = CachedBlackBoxTarget(
            prob_matrix=ev.hc_probs.mean(axis=0),
            target_id="target_0",
            query_budget=5,
            query_indices=list(range(30)),
        )
        # Any query that raises estimated leakage above 0.50 should fail feasibility
        # If all candidates raise leakage, it must raise NoFeasibleQueriesError
        try:
            q1 = pol.select_next(t.transcript)
            t.query(q1)
            # After querying, leakage will be >= 0.50
            with pytest.raises(NoFeasibleQueriesError):
                pol.select_next(t.transcript)
        except NoFeasibleQueriesError:
            pass  # Expected behavior when budget is immediately binding


# ---------------------------------------------------------------------------
# Test Phase 14: Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_temperature_calibrator_fit(self):
        rng = np.random.default_rng(42)
        # Generate overconfident LLRs
        y = np.array([1] * 20 + [0] * 20)
        llrs = np.concatenate([
            rng.normal(10.0, 1.0, size=20),
            rng.normal(-10.0, 1.0, size=20),
        ])
        cal = TemperatureCalibrator()
        cal.fit(llrs, y)
        assert cal.temperature > 0.1
        calibrated_llrs = cal.calibrate_llr(llrs)
        assert len(calibrated_llrs) == 40
        probs = cal.calibrate_prob(llrs)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_threshold_calibrator(self):
        llrs = [5.0, 4.0, 3.5, 2.0, -1.0, -3.0, -4.5, -6.0]
        y = [1, 1, 1, 1, 0, 0, 0, 0]
        cal = ThresholdCalibrator(target_false_pass=0.05, target_false_fail=0.05)
        thresh = cal.calibrate_thresholds(llrs, y)
        assert thresh.theta_pass > 0.0
        assert thresh.theta_fail < 0.0
        assert thresh.false_pass_rate <= 0.05
        assert thresh.false_fail_rate <= 0.05
        d = thresh.to_dict()
        assert "theta_pass" in d
        assert "theta_fail" in d


# ---------------------------------------------------------------------------
# Test Phase 15: Sequential Stopping Engine
# ---------------------------------------------------------------------------

class TestSequentialStoppingEngine:
    def test_engine_stops_on_pass_for_compliant_target(self):
        ev = _make_synthetic_surrogate()
        # Target matching HC mean exactly
        hc_mean = ev.hc_probs.mean(axis=0)
        t = CachedBlackBoxTarget(
            prob_matrix=hc_mean,
            target_id="compliant_target",
            query_budget=20,
            query_indices=list(range(30)),
        )
        pol = PRISMUPolicy(pool_size=30, evidence_model=ev)
        engine = SequentialStoppingEngine(
            evidence_model=ev,
            config=StoppingConfig(theta_pass=2.0, theta_fail=-2.0, query_budget=20),
        )
        res = engine.run_audit(target=t, policy=pol)
        assert res.verdict == "PASS"
        assert res.reason == "threshold_pass"
        assert res.queries_used <= 20
        assert res.final_llr >= 2.0
        assert res.final_p_compliant > 0.5

    def test_engine_stops_on_fail_for_noncompliant_target(self):
        ev = _make_synthetic_surrogate()
        # Target matching HN mean exactly
        hn_mean = ev.hn_probs.mean(axis=0)
        t = CachedBlackBoxTarget(
            prob_matrix=hn_mean,
            target_id="noncompliant_target",
            query_budget=20,
            query_indices=list(range(30)),
        )
        pol = PRISMUPolicy(pool_size=30, evidence_model=ev)
        engine = SequentialStoppingEngine(
            evidence_model=ev,
            config=StoppingConfig(theta_pass=2.0, theta_fail=-2.0, query_budget=20),
        )
        res = engine.run_audit(target=t, policy=pol)
        assert res.verdict == "FAIL"
        assert res.reason == "threshold_fail"
        assert res.queries_used <= 20
        assert res.final_llr <= -2.0
        assert res.final_p_compliant < 0.5

    def test_engine_returns_inconclusive_when_budget_exhausted(self):
        ev = _make_synthetic_surrogate()
        # Ambiguous target (equal mixture of HC and HN)
        ambiguous = 0.5 * (ev.hc_probs.mean(axis=0) + ev.hn_probs.mean(axis=0))
        t = CachedBlackBoxTarget(
            prob_matrix=ambiguous,
            target_id="ambiguous_target",
            query_budget=3,
            query_indices=list(range(30)),
        )
        pol = PRISMUPolicy(pool_size=30, evidence_model=ev)
        engine = SequentialStoppingEngine(
            evidence_model=ev,
            # Unattainable thresholds within 3 queries
            config=StoppingConfig(theta_pass=50.0, theta_fail=-50.0, query_budget=3),
        )
        res = engine.run_audit(target=t, policy=pol)
        assert res.verdict == "INCONCLUSIVE"
        assert res.reason == "query_budget_exhausted"
        assert res.queries_used == 3

    def test_engine_stops_when_privacy_budget_exhausted(self):
        ev = _make_synthetic_surrogate()
        hc_mean = ev.hc_probs.mean(axis=0)
        t = CachedBlackBoxTarget(
            prob_matrix=hc_mean,
            target_id="privacy_limited_target",
            query_budget=20,
            query_indices=list(range(30)),
        )
        # Bind privacy budget tightly so policy cannot continue
        pol = PRISMUPolicy(pool_size=30, evidence_model=ev, privacy_budget=0.51)
        engine = SequentialStoppingEngine(
            evidence_model=ev,
            config=StoppingConfig(
                theta_pass=50.0,  # Ensure no early PASS
                theta_fail=-50.0,
                query_budget=20,
                privacy_budget=0.51,
            ),
        )
        res = engine.run_audit(target=t, policy=pol)
        assert res.verdict == "INCONCLUSIVE"
        assert "privacy_budget_exhausted" in res.reason

    def test_engine_serialization(self):
        ev = _make_synthetic_surrogate()
        hc_mean = ev.hc_probs.mean(axis=0)
        t = CachedBlackBoxTarget(
            prob_matrix=hc_mean,
            target_id="target_ser",
            query_budget=5,
            query_indices=list(range(30)),
        )
        pol = PRISMUPolicy(pool_size=30, evidence_model=ev)
        engine = SequentialStoppingEngine(
            evidence_model=ev,
            config=StoppingConfig(theta_pass=1.0, theta_fail=-1.0, query_budget=5),
        )
        res = engine.run_audit(target=t, policy=pol)
        d = res.to_dict()
        assert d["target_id"] == "target_ser"
        assert "verdict" in d
        assert "queries_used" in d
        assert "steps" in d
