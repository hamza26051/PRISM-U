"""Unit tests for Phase 10 (MIA attacks) and Phase 11 (privacy risk estimation)."""
from __future__ import annotations

import numpy as np
import pytest

from attacks.membership_inference import (
    ALL_ATTACKS,
    ConfidenceAttack,
    EntropyAttack,
    LogisticRegressionAttack,
    LossAttack,
    MarginAttack,
    MLPAttack,
)
from attacks.attack_evaluation import (
    TranscriptLeakageResult,
    bootstrap_mia_auroc_ci,
    compute_mia_advantage,
    compute_mia_auroc,
    evaluate_all_attacks,
    evaluate_single_attack,
    evaluate_transcript_leakage,
)
from audit.privacy_risk import StaticQueryRisk, TranscriptPrivacyEstimator


# ---------------------------------------------------------------------------
# Fixtures: Synthetic surrogate banks with known member/non-member separation
# ---------------------------------------------------------------------------

def _make_separable_data(seed: int = 42):
    """Create synthetic probability vectors where members and non-members differ.

    Members (retained data): model outputs higher confidence on correct class.
    Non-members: model outputs more uniform predictions.
    """
    rng = np.random.default_rng(seed)
    n_mem, n_non, k = 200, 200, 10

    # Members: concentrated on one class (simulating memorization)
    member_probs = np.full((n_mem, k), 0.01, dtype=np.float64)
    for i in range(n_mem):
        c = rng.integers(0, k)
        member_probs[i, c] = 0.80 + rng.uniform(0, 0.10)
    member_probs /= member_probs.sum(axis=1, keepdims=True)
    member_labels = np.argmax(member_probs, axis=1)

    # Non-members: more uniform predictions
    nonmember_probs = np.full((n_non, k), 0.05, dtype=np.float64)
    for i in range(n_non):
        c = rng.integers(0, k)
        nonmember_probs[i, c] = 0.40 + rng.uniform(0, 0.15)
    nonmember_probs /= nonmember_probs.sum(axis=1, keepdims=True)
    nonmember_labels = np.argmax(nonmember_probs, axis=1)

    return member_probs, nonmember_probs, member_labels, nonmember_labels


def _make_indistinguishable_data(seed: int = 99):
    """Create synthetic data where members and non-members are identical (no leakage)."""
    rng = np.random.default_rng(seed)
    n, k = 100, 10
    probs = rng.dirichlet(np.ones(k), size=2 * n).astype(np.float64)
    labels = np.argmax(probs, axis=1)
    return probs[:n], probs[n:], labels[:n], labels[n:]


def _make_surrogate_bank(seed: int = 42):
    """Create synthetic HC/HN surrogate banks with distinguishable behavior.

    HC (retrained on D_r only): lower confidence, higher entropy on some queries.
    HN (trained on D_r ∪ D_u): higher confidence, lower entropy on those queries.
    """
    rng = np.random.default_rng(seed)
    n_hc, n_hn, n_q, k = 12, 48, 50, 10

    # Base uniform responses
    hc_probs = rng.dirichlet(np.ones(k) * 5, size=(n_hc, n_q)).astype(np.float64)
    hn_probs = rng.dirichlet(np.ones(k) * 5, size=(n_hn, n_q)).astype(np.float64)

    # Make some queries discriminative (HC vs HN differ on queries 0-9)
    for q in range(10):
        # HC: more uniform on these queries
        hc_probs[:, q, :] = rng.dirichlet(np.ones(k) * 3, size=n_hc)
        # HN: more peaked on these queries (retaining forget data influence)
        peaked = np.full((n_hn, k), 0.02)
        for i in range(n_hn):
            peaked[i, q % k] = 0.75 + rng.uniform(0, 0.05)
        peaked /= peaked.sum(axis=1, keepdims=True)
        hn_probs[:, q, :] = peaked

    return hc_probs, hn_probs


# ---------------------------------------------------------------------------
# Phase 10 Tests: Membership Inference Attacks
# ---------------------------------------------------------------------------

class TestMetricAttacks:
    """Test metric-based MIA attacks."""

    def test_loss_attack_produces_valid_scores(self):
        mem_p, non_p, mem_l, non_l = _make_separable_data()
        atk = LossAttack()
        all_p = np.vstack([mem_p, non_p])
        all_l = np.concatenate([mem_l, non_l])
        output = atk.attack(all_p, labels=all_l)
        assert output.scores.shape == (400,)
        assert output.attack_name == "loss"
        # Members should have higher scores (lower loss) on average
        mem_mean = output.scores[:200].mean()
        non_mean = output.scores[200:].mean()
        assert mem_mean > non_mean

    def test_loss_attack_requires_labels(self):
        mem_p, _, _, _ = _make_separable_data()
        atk = LossAttack()
        with pytest.raises(ValueError, match="requires ground-truth labels"):
            atk.attack(mem_p, labels=None)

    def test_entropy_attack_scores(self):
        mem_p, non_p, _, _ = _make_separable_data()
        atk = EntropyAttack()
        all_p = np.vstack([mem_p, non_p])
        output = atk.attack(all_p)
        assert output.scores.shape == (400,)
        # Members have lower entropy → higher scores
        mem_mean = output.scores[:200].mean()
        non_mean = output.scores[200:].mean()
        assert mem_mean > non_mean

    def test_confidence_attack_scores(self):
        mem_p, non_p, _, _ = _make_separable_data()
        atk = ConfidenceAttack()
        all_p = np.vstack([mem_p, non_p])
        output = atk.attack(all_p)
        assert output.scores.shape == (400,)
        # Members have higher confidence
        assert output.scores[:200].mean() > output.scores[200:].mean()

    def test_margin_attack_scores(self):
        mem_p, non_p, _, _ = _make_separable_data()
        atk = MarginAttack()
        all_p = np.vstack([mem_p, non_p])
        output = atk.attack(all_p)
        assert output.scores.shape == (400,)
        # Members have larger margin
        assert output.scores[:200].mean() > output.scores[200:].mean()


class TestLearnedAttacks:
    """Test learned MIA attacks."""

    def test_logistic_regression_fit_and_attack(self):
        mem_p, non_p, _, _ = _make_separable_data(seed=1)
        atk = LogisticRegressionAttack(C=1.0, seed=42)
        atk.fit(mem_p[:100], non_p[:100])
        output = atk.attack(np.vstack([mem_p[100:], non_p[100:]]))
        assert output.scores.shape == (200,)
        assert output.predictions is not None
        assert output.predictions.shape == (200,)
        assert set(output.predictions.tolist()).issubset({0, 1})

    def test_logistic_regression_requires_fit(self):
        atk = LogisticRegressionAttack()
        with pytest.raises(RuntimeError, match="must be fit"):
            atk.attack(np.random.rand(10, 10))

    def test_mlp_fit_and_attack(self):
        mem_p, non_p, _, _ = _make_separable_data(seed=2)
        atk = MLPAttack(hidden_sizes=(32, 16), max_iter=100, seed=42)
        atk.fit(mem_p[:100], non_p[:100])
        output = atk.attack(np.vstack([mem_p[100:], non_p[100:]]))
        assert output.scores.shape == (200,)
        assert output.predictions is not None

    def test_mlp_requires_fit(self):
        atk = MLPAttack()
        with pytest.raises(RuntimeError, match="must be fit"):
            atk.attack(np.random.rand(10, 10))


class TestAttackEvaluation:
    """Test the attack evaluation framework."""

    def test_mia_auroc_separable(self):
        mem_p, non_p, _, _ = _make_separable_data()
        atk = ConfidenceAttack()
        all_p = np.vstack([mem_p, non_p])
        output = atk.attack(all_p)
        labels = np.concatenate([np.ones(200), np.zeros(200)])
        auroc = compute_mia_auroc(output.scores, labels)
        assert auroc > 0.7  # Should be well above random

    def test_mia_auroc_indistinguishable(self):
        mem_p, non_p, _, _ = _make_indistinguishable_data()
        atk = ConfidenceAttack()
        all_p = np.vstack([mem_p, non_p])
        output = atk.attack(all_p)
        labels = np.concatenate([np.ones(100), np.zeros(100)])
        auroc = compute_mia_auroc(output.scores, labels)
        assert 0.3 < auroc < 0.7  # Should be near random

    def test_mia_advantage_range(self):
        assert compute_mia_advantage(0.5) == 0.0
        assert compute_mia_advantage(1.0) == 1.0
        assert compute_mia_advantage(0.75) == pytest.approx(0.5)
        assert compute_mia_advantage(0.3) == 0.0  # Clamped

    def test_bootstrap_ci_coverage(self):
        mem_p, non_p, _, _ = _make_separable_data()
        atk = ConfidenceAttack()
        all_p = np.vstack([mem_p, non_p])
        output = atk.attack(all_p)
        labels = np.concatenate([np.ones(200), np.zeros(200)])
        point, lo, hi = bootstrap_mia_auroc_ci(output.scores, labels, n_boot=200)
        assert lo <= point <= hi
        assert lo > 0.0
        assert hi <= 1.0

    def test_evaluate_single_attack(self):
        mem_p, non_p, _, _ = _make_separable_data()
        atk = ConfidenceAttack()
        result = evaluate_single_attack(
            atk,
            member_probs=mem_p,
            nonmember_probs=non_p,
            n_boot=100,
        )
        assert result.attack_name == "confidence"
        assert result.auroc > 0.6
        assert result.advantage >= 0.0
        assert result.n_members == 200
        assert result.n_nonmembers == 200
        assert result.ci_low is not None
        assert result.ci_high is not None

    def test_evaluate_all_attacks_full_pipeline(self):
        mem_p, non_p, mem_l, non_l = _make_separable_data()
        results = evaluate_all_attacks(
            member_probs_train=mem_p[:100],
            nonmember_probs_train=non_p[:100],
            member_probs_eval=mem_p[100:],
            nonmember_probs_eval=non_p[100:],
            member_labels_train=mem_l[:100],
            nonmember_labels_train=non_l[:100],
            member_labels_eval=mem_l[100:],
            nonmember_labels_eval=non_l[100:],
            n_boot=50,
        )
        # Should have 4 metric + 2 learned = 6 attacks
        assert len(results) == 6
        names = [r.attack_name for r in results]
        assert "loss" in names
        assert "entropy" in names
        assert "confidence" in names
        assert "margin" in names
        assert "logistic_regression" in names
        assert "mlp" in names
        # All results should have valid metrics
        for r in results:
            assert 0.0 <= r.auroc <= 1.0
            assert 0.0 <= r.advantage <= 1.0


class TestTranscriptLeakage:
    """Test transcript-based leakage evaluation."""

    def test_transcript_leakage_with_discriminative_queries(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        surr_probs = np.concatenate([hc_probs, hn_probs], axis=0)
        hyps = np.array(["HC"] * 12 + ["HN"] * 48)

        # Query discriminative queries (0-9)
        result = evaluate_transcript_leakage(
            transcript_responses=hc_probs[0, :10, :],
            transcript_query_ids=list(range(10)),
            surrogate_probs=surr_probs,
            surrogate_hypotheses=hyps,
            target_id="test_target",
            n_boot=50,
        )
        assert isinstance(result, TranscriptLeakageResult)
        assert result.n_queries == 10
        assert result.max_auroc > 0.5  # Should detect some leakage
        assert len(result.attack_results) > 0

    def test_transcript_leakage_empty_transcript(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        surr_probs = np.concatenate([hc_probs, hn_probs], axis=0)
        hyps = np.array(["HC"] * 12 + ["HN"] * 48)

        result = evaluate_transcript_leakage(
            transcript_responses=np.empty((0, 10)),
            transcript_query_ids=[],
            surrogate_probs=surr_probs,
            surrogate_hypotheses=hyps,
        )
        assert result.n_queries == 0
        assert result.max_auroc == 0.5
        assert result.max_advantage == 0.0

    def test_transcript_leakage_serialization(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        surr_probs = np.concatenate([hc_probs, hn_probs], axis=0)
        hyps = np.array(["HC"] * 12 + ["HN"] * 48)

        result = evaluate_transcript_leakage(
            transcript_responses=hc_probs[0, :5, :],
            transcript_query_ids=list(range(5)),
            surrogate_probs=surr_probs,
            surrogate_hypotheses=hyps,
            n_boot=20,
        )
        d = result.to_dict()
        assert "target_id" in d
        assert "max_auroc" in d
        assert "attacks" in d


# ---------------------------------------------------------------------------
# Phase 11 Tests: Privacy Risk Estimation
# ---------------------------------------------------------------------------

class TestStaticQueryRisk:
    """Test static per-query risk estimation."""

    def test_static_risk_shape_and_range(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        risk = StaticQueryRisk(hc_probs=hc_probs, hn_probs=hn_probs)
        all_risks = risk.all_risks
        assert all_risks.shape == (50,)
        assert np.all(all_risks >= 0.5)
        assert np.all(all_risks <= 1.0)

    def test_discriminative_queries_have_higher_risk(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        risk = StaticQueryRisk(hc_probs=hc_probs, hn_probs=hn_probs)
        # Queries 0-9 were made discriminative in _make_surrogate_bank
        disc_risk = risk.score_candidates(list(range(10))).mean()
        nondisc_risk = risk.score_candidates(list(range(10, 50))).mean()
        assert disc_risk > nondisc_risk

    def test_score_candidates_subset(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        risk = StaticQueryRisk(hc_probs=hc_probs, hn_probs=hn_probs)
        scores = risk.score_candidates([0, 5, 25, 49])
        assert scores.shape == (4,)
        assert np.all(scores >= 0.5)

    def test_from_evidence_model(self):
        from audit.evidence import SurrogateEvidenceModel
        hc_probs, hn_probs = _make_surrogate_bank()
        ev = SurrogateEvidenceModel(hc_probs=hc_probs, hn_probs=hn_probs, bandwidth=1.0)
        risk = StaticQueryRisk.from_evidence_model(ev)
        assert risk.all_risks.shape == (50,)


class TestTranscriptPrivacyEstimator:
    """Test transcript-dependent privacy risk estimation."""

    def test_cumulative_leakage_empty_transcript(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        estimator = TranscriptPrivacyEstimator(
            hc_probs=hc_probs, hn_probs=hn_probs, fast_mode=True,
        )
        leakage = estimator.cumulative_leakage([])
        assert leakage == 0.5  # No queries → no leakage

    def test_cumulative_leakage_increases_with_queries(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        estimator = TranscriptPrivacyEstimator(
            hc_probs=hc_probs, hn_probs=hn_probs, fast_mode=True,
        )
        # Leakage with 1 discriminative query
        leak_1 = estimator.cumulative_leakage([0])
        # Leakage with 5 discriminative queries
        leak_5 = estimator.cumulative_leakage([0, 1, 2, 3, 4])
        # More queries should give at least as much leakage
        assert leak_5 >= leak_1

    def test_marginal_risk_shape(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        estimator = TranscriptPrivacyEstimator(
            hc_probs=hc_probs, hn_probs=hn_probs, fast_mode=True,
        )
        risks = estimator.marginal_risk(
            query_ids=[0, 1],
            candidate_ids=[2, 3, 4, 5],
        )
        assert risks.shape == (4,)

    def test_marginal_risk_discriminative_vs_nondiscriminative(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        estimator = TranscriptPrivacyEstimator(
            hc_probs=hc_probs, hn_probs=hn_probs, fast_mode=True,
        )
        # Marginal risk of discriminative query vs non-discriminative
        risk_disc = estimator.marginal_risk(query_ids=[], candidate_ids=[0])
        risk_nondisc = estimator.marginal_risk(query_ids=[], candidate_ids=[30])
        # Discriminative queries should have higher marginal risk
        assert risk_disc[0] >= risk_nondisc[0]

    def test_marginal_risk_empty_candidates(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        estimator = TranscriptPrivacyEstimator(
            hc_probs=hc_probs, hn_probs=hn_probs, fast_mode=True,
        )
        risks = estimator.marginal_risk(query_ids=[0], candidate_ids=[])
        assert risks.shape == (0,)

    def test_from_evidence_model_constructor(self):
        from audit.evidence import SurrogateEvidenceModel
        hc_probs, hn_probs = _make_surrogate_bank()
        ev = SurrogateEvidenceModel(hc_probs=hc_probs, hn_probs=hn_probs, bandwidth=1.0)
        estimator = TranscriptPrivacyEstimator.from_evidence_model(ev)
        leakage = estimator.cumulative_leakage([0, 1, 2])
        assert 0.5 <= leakage <= 1.0

    def test_batched_marginal_risk(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        estimator = TranscriptPrivacyEstimator(
            hc_probs=hc_probs, hn_probs=hn_probs, fast_mode=True,
        )
        # With top_k filtering
        risks = estimator.marginal_risk_batched(
            query_ids=[],
            candidate_ids=list(range(50)),
            top_k=10,
        )
        assert risks.shape == (50,)

    def test_static_risk_property(self):
        hc_probs, hn_probs = _make_surrogate_bank()
        estimator = TranscriptPrivacyEstimator(
            hc_probs=hc_probs, hn_probs=hn_probs, fast_mode=True,
        )
        assert estimator.static_risk is not None
        assert estimator.static_risk.all_risks.shape == (50,)
