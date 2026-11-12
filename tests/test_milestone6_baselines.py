"""Unit tests for Milestone 6 auditing baselines, evidence updates, and metrics."""
from __future__ import annotations

import numpy as np
import pytest

from audit.audit_gain import AdaptivePosteriorGainEstimator, PrivacyBlindActivePolicy, StaticJSGainEstimator
from audit.baseline_auditors import run_baseline_audit
from audit.evidence import SurrogateEvidenceModel
from audit.fixed_policy import FixedTopKPolicy
from audit.random_policy import RandomQueryPolicy
from evaluation.metrics import bootstrap_auroc_ci, compute_auroc, compute_brier_score, compute_decision_metrics
from models.black_box_target import CachedBlackBoxTarget


def _make_synthetic_surrogate():
    """Create synthetic surrogate bank [M, Q=8, K=3] with known informative vs uninformative queries."""
    rng = np.random.default_rng(7)
    n_hc, n_hn, n_q, n_c = 6, 12, 8, 3
    hc = np.full((n_hc, n_q, n_c), 1.0 / n_c, dtype=np.float64)
    hn = np.full((n_hn, n_q, n_c), 1.0 / n_c, dtype=np.float64)

    # Queries 0, 1: uninformative noise
    hc[:, 0:2, :] = np.array([0.34, 0.33, 0.33])
    hn[:, 0:2, :] = np.array([0.33, 0.34, 0.33])

    # Query 2: informative (HC -> class 0, HN -> class 1)
    hc[:, 2, :] = np.array([0.85, 0.10, 0.05])
    hn[:, 2, :] = np.array([0.10, 0.85, 0.05])

    # Queries 3, 4: conditional sub-structure! Half of HC+HN agree on q=3, disagree on q=4 and vice-versa
    hc[:3, 3, :] = np.array([0.90, 0.05, 0.05])
    hc[3:, 3, :] = np.array([0.10, 0.80, 0.10])
    hn[:6, 3, :] = np.array([0.10, 0.80, 0.10])
    hn[6:, 3, :] = np.array([0.90, 0.05, 0.05])

    hc[:3, 4, :] = np.array([0.85, 0.10, 0.05])
    hc[3:, 4, :] = np.array([0.50, 0.25, 0.25])
    hn[:6, 4, :] = np.array([0.50, 0.25, 0.25])
    hn[6:, 4, :] = np.array([0.10, 0.10, 0.80])

    # Query 5: moderately informative
    hc[:, 5, :] = np.array([0.70, 0.20, 0.10])
    hn[:, 5, :] = np.array([0.20, 0.70, 0.10])

    # Add tiny jitter and renormalize
    hc = np.clip(hc + rng.normal(0, 0.01, size=hc.shape), 1e-4, 1.0)
    hn = np.clip(hn + rng.normal(0, 0.01, size=hn.shape), 1e-4, 1.0)
    hc /= hc.sum(axis=-1, keepdims=True)
    hn /= hn.sum(axis=-1, keepdims=True)

    return SurrogateEvidenceModel(hc_probs=hc, hn_probs=hn, bandwidth=0.5, access_mode="probabilities")


def test_evidence_sign_convention_and_policies():
    ev = _make_synthetic_surrogate()
    gain_est = AdaptivePosteriorGainEstimator(ev)
    static_est = StaticJSGainEstimator(ev)
    assert static_est.score_candidates(ev.initial_state(), [0, 2])[1] > static_est.score_candidates(ev.initial_state(), [0, 2])[0]

    # Compliant target matching HC mean
    hc_target_probs = ev.hc_probs.mean(axis=0)
    hn_target_probs = ev.hn_probs.mean(axis=0)
    q_idx = list(range(8))

    t_hc = CachedBlackBoxTarget(prob_matrix=hc_target_probs, target_id="hc", query_budget=5, query_indices=q_idx)
    t_hn = CachedBlackBoxTarget(prob_matrix=hn_target_probs, target_id="hn", query_budget=5, query_indices=q_idx)

    active = PrivacyBlindActivePolicy(pool_size=8, evidence_model=ev, gain_estimator=gain_est)
    res_hc = run_baseline_audit(target=t_hc, policy=active, evidence_model=ev, max_queries=5)
    res_hn = run_baseline_audit(target=t_hn, policy=active, evidence_model=ev, max_queries=5)

    # Positive LLR for compliant, negative LLR for non-compliant
    assert res_hc.final_llr > 0.0
    assert res_hc.final_p_compliant > 0.5
    assert res_hn.final_llr < 0.0
    assert res_hn.final_p_compliant < 0.5

    # No duplicate queries
    assert len(set(t_hc.transcript.query_ids)) == 5
    assert len(set(t_hn.transcript.query_ids)) == 5

    # Random and Fixed policies obey non-repetition & budget
    rand_pol = RandomQueryPolicy(pool_size=8, seed=123)
    t_rand = CachedBlackBoxTarget(prob_matrix=hc_target_probs, target_id="r", query_budget=5, query_indices=q_idx)
    run_baseline_audit(target=t_rand, policy=rand_pol, evidence_model=ev, max_queries=5)
    assert len(set(t_rand.transcript.query_ids)) == 5

    fixed_pol = FixedTopKPolicy(ranked_query_ids=[2, 5, 4, 3, 0, 1, 6, 7])
    t_fix = CachedBlackBoxTarget(prob_matrix=hc_target_probs, target_id="f", query_budget=5, query_indices=q_idx)
    res_fix = run_baseline_audit(target=t_fix, policy=fixed_pol, evidence_model=ev, max_queries=5)
    assert [s.query_id for s in res_fix.steps] == [2, 5, 4, 3, 0]


def test_adaptive_gain_changes_conditioned_on_transcript():
    ev = _make_synthetic_surrogate()
    gain_est = AdaptivePosteriorGainEstimator(ev)
    s0 = ev.initial_state()

    # After observing two distinctly opposite responses at query 3, conditional gains over remaining queries must diverge
    s_branch_a = ev.update(s0, 3, ev.hc_probs[0, 3])
    s_branch_b = ev.update(s0, 3, ev.hc_probs[4, 3])

    g_a = gain_est.score_candidates(s_branch_a, [0, 1, 2, 4, 5, 6, 7])
    g_b = gain_est.score_candidates(s_branch_b, [0, 1, 2, 4, 5, 6, 7])
    assert not np.allclose(g_a, g_b, atol=1e-5)


def test_evaluation_metrics():
    scores = [2.5, 1.2, -0.5, -2.0]
    probs = [0.92, 0.77, 0.38, 0.12]
    y = [1, 1, 0, 0]
    assert compute_auroc(scores, y) == 1.0
    assert 0.0 < compute_brier_score(probs, y) < 0.1

    ci = bootstrap_auroc_ci(scores, y, n_boot=200, seed=42)
    assert ci["auroc"] == 1.0
    assert 0.0 <= ci["ci_low"] <= ci["ci_high"] <= 1.0

    dec = compute_decision_metrics(["PASS", "INCONCLUSIVE", "PASS", "FAIL"], y)
    assert dec["coverage"] == 0.75
    assert dec["inconclusive_rate"] == 0.25
    assert dec["false_pass_rate"] == 0.5
    assert dec["false_fail_rate"] == 0.0
    assert dec["conditional_accuracy"] == pytest.approx(2.0 / 3.0)

