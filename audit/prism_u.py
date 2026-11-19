r"""PRISM-U joint acquisition policies under query and privacy budgets.

Implements:
1. StaticPrivacyAwarePolicy: Static gain G(x) - lambda * R(x).
2. AdaptiveGainStaticPrivacyPolicy: Adaptive gain G_t(x | tau_t) - lambda * R(x).
3. PRISMUPolicy: Full PRISM-U joint policy with:
   - Response-adaptive marginal compliance gain G_t(x | tau_t)
   - Transcript-dependent marginal privacy risk Delta L_t(x | tau_t)
   - Hard cumulative privacy budget feasibility filtering:
     Feasible_t = { x in Q \ tau_t | L_hat(tau_t) + max(0, Delta L_t(x)) <= B_P }
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from audit.acquisition import AcquisitionScorer, AdditiveAcquisitionScorer, RatioAcquisitionScorer
from audit.audit_gain import AdaptivePosteriorGainEstimator, StaticJSGainEstimator
from audit.evidence import EvidenceState, SurrogateEvidenceModel
from audit.privacy_risk import StaticQueryRisk, TranscriptPrivacyEstimator
from audit.transcript import AuditTranscript


class NoFeasibleQueriesError(RuntimeError):
    """Raised when no candidate query satisfies the privacy budget constraint."""


class StaticPrivacyAwarePolicy:
    """Non-adaptive privacy-aware baseline: scores candidates using static gain and static risk.

    Score(x) = G_static(x) - lambda * (R_static(x) - 0.5)
    """

    name = "static_privacy_aware"

    def __init__(
        self,
        pool_size: int,
        evidence_model: SurrogateEvidenceModel,
        penalty_lambda: float = 1.0,
        static_gain: Optional[StaticJSGainEstimator] = None,
        static_risk: Optional[StaticQueryRisk] = None,
    ) -> None:
        self.pool_size = int(pool_size)
        self.penalty_lambda = float(penalty_lambda)
        self.evidence_model = evidence_model

        self.static_gain = static_gain or StaticJSGainEstimator(evidence_model)
        self.static_risk = static_risk or StaticQueryRisk.from_evidence_model(evidence_model)

        # Precompute static scores across all pool indices
        init_state = evidence_model.initial_state()
        all_ids = list(range(self.pool_size))
        g = self.static_gain.score_candidates(init_state, all_ids)
        r = self.static_risk.score_candidates(all_ids)
        # Shift risk by 0.5 so non-leaking queries have risk 0
        self._static_scores = g - self.penalty_lambda * (r - 0.5)
        self._last_scores: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last_scores = None

    def select_next(
        self,
        transcript: AuditTranscript,
        state: Optional[EvidenceState] = None,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> int:
        queried = transcript.queried_id_set
        pool = range(self.pool_size) if candidate_ids is None else candidate_ids
        available = [int(q) for q in pool if int(q) not in queried]
        if not available:
            raise RuntimeError("No unqueried candidates remain in pool.")

        avail_scores = self._static_scores[available]
        self._last_scores = avail_scores
        best_idx = int(np.argmax(avail_scores))
        return available[best_idx]


class AdaptiveGainStaticPrivacyPolicy:
    """Semi-adaptive baseline: adaptive compliance gain G_t(x | tau_t) with static risk R(x).

    Score(x) = G_t(x | tau_t) - lambda * (R_static(x) - 0.5)
    """

    name = "adaptive_gain_static_privacy"

    def __init__(
        self,
        pool_size: int,
        evidence_model: SurrogateEvidenceModel,
        penalty_lambda: float = 1.0,
        gain_estimator: Optional[AdaptivePosteriorGainEstimator] = None,
        static_risk: Optional[StaticQueryRisk] = None,
        scorer: Optional[AcquisitionScorer] = None,
    ) -> None:
        self.pool_size = int(pool_size)
        self.penalty_lambda = float(penalty_lambda)
        self.evidence_model = evidence_model

        self.gain_estimator = gain_estimator or AdaptivePosteriorGainEstimator(evidence_model)
        self.static_risk = static_risk or StaticQueryRisk.from_evidence_model(evidence_model)
        self.scorer = scorer or AdditiveAcquisitionScorer(penalty_lambda=self.penalty_lambda)
        self._last_scores: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last_scores = None

    def select_next(
        self,
        transcript: AuditTranscript,
        state: Optional[EvidenceState] = None,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> int:
        queried = transcript.queried_id_set
        pool = range(self.pool_size) if candidate_ids is None else candidate_ids
        available = [int(q) for q in pool if int(q) not in queried]
        if not available:
            raise RuntimeError("No unqueried candidates remain in pool.")

        if state is None:
            state = self.evidence_model.score_transcript(transcript.query_ids, transcript.responses)

        gains = self.gain_estimator.score_candidates(state, available)
        risks = self.static_risk.score_candidates(available) - 0.5  # Zero-baseline risk

        scores = self.scorer.score(gains, risks, step=len(transcript))
        self._last_scores = scores
        best_idx = int(np.argmax(scores))
        return available[best_idx]


class PRISMUPolicy:
    r"""Full PRISM-U joint active auditing acquisition policy.

    Selects queries optimizing information gain under empirical privacy expenditure
    and hard cumulative privacy budget feasibility filtering:

        Feasible_t = { x in Q \ tau_t | L_hat(tau_t) + max(0, Delta L_t(x | tau_t)) <= B_P }
        x*_t = argmax_{x in Feasible_t} Scorer(G_t(x | tau_t), Delta L_t(x | tau_t))
    """

    name = "prism_u"

    def __init__(
        self,
        pool_size: int,
        evidence_model: SurrogateEvidenceModel,
        privacy_budget: float = float("inf"),
        privacy_estimator: Optional[TranscriptPrivacyEstimator] = None,
        gain_estimator: Optional[AdaptivePosteriorGainEstimator] = None,
        scorer: Optional[AcquisitionScorer] = None,
        penalty_lambda: float = 1.0,
        acquisition_mode: str = "additive",
        pre_filter_k: Optional[int] = 50,
    ) -> None:
        """
        Args:
            pool_size: Number of queries in auxiliary pool Q.
            evidence_model: Calibrated SurrogateEvidenceModel.
            privacy_budget: Hard privacy expenditure cap B_P in [0.5, 1.0] (or inf for unconstrained).
            privacy_estimator: TranscriptPrivacyEstimator instance.
            gain_estimator: AdaptivePosteriorGainEstimator instance.
            scorer: Custom AcquisitionScorer (overrides acquisition_mode if provided).
            penalty_lambda: Weight on privacy risk for additive acquisition.
            acquisition_mode: "additive" or "ratio".
            pre_filter_k: Number of candidate queries to evaluate for full marginal risk.
                          If set, ranks unqueried candidates by gain, and evaluates marginal
                          risk on top-k candidates to accelerate online selection.
        """
        self.pool_size = int(pool_size)
        self.evidence_model = evidence_model
        self.privacy_budget = float(privacy_budget)
        self.pre_filter_k = None if pre_filter_k is None else int(pre_filter_k)

        self.gain_estimator = gain_estimator or AdaptivePosteriorGainEstimator(evidence_model)
        self.privacy_estimator = privacy_estimator or TranscriptPrivacyEstimator.from_evidence_model(
            evidence_model, fast_mode=True,
        )

        if scorer is not None:
            self.scorer = scorer
        elif acquisition_mode == "ratio":
            self.scorer = RatioAcquisitionScorer(eps=1e-4)
        elif acquisition_mode == "additive":
            self.scorer = AdditiveAcquisitionScorer(penalty_lambda=penalty_lambda)
        else:
            raise ValueError(f"Unknown acquisition_mode: {acquisition_mode}")

        self._last_scores: Optional[np.ndarray] = None
        self._last_feasible_count: int = 0
        self._current_leakage: float = 0.5

    def reset(self) -> None:
        self._last_scores = None
        self._last_feasible_count = 0
        self._current_leakage = 0.5

    @property
    def current_estimated_leakage(self) -> float:
        return self._current_leakage

    def select_next(
        self,
        transcript: AuditTranscript,
        state: Optional[EvidenceState] = None,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> int:
        queried = transcript.queried_id_set
        pool = range(self.pool_size) if candidate_ids is None else candidate_ids
        available = [int(q) for q in pool if int(q) not in queried]
        if not available:
            raise RuntimeError("No unqueried candidates remain in pool.")

        if state is None:
            state = self.evidence_model.score_transcript(transcript.query_ids, transcript.responses)

        # 1. Compute current cumulative leakage
        current_qids = transcript.query_ids
        cur_leakage = self.privacy_estimator.cumulative_leakage(current_qids)
        self._current_leakage = cur_leakage

        # Check if privacy budget already exhausted
        if np.isfinite(self.privacy_budget) and cur_leakage >= self.privacy_budget:
            raise NoFeasibleQueriesError(
                f"Privacy budget B_P={self.privacy_budget:.3f} already exhausted by transcript (L_hat={cur_leakage:.3f})."
            )

        # 2. Compute compliance information gain for all available candidates
        gains = self.gain_estimator.score_candidates(state, available)

        # 3. Acceleration: if pre_filter_k is set and available > pre_filter_k,
        # evaluate prospective marginal risk only on top-k candidates by gain
        if self.pre_filter_k is not None and len(available) > self.pre_filter_k:
            top_gain_idx = np.argsort(-gains)[: self.pre_filter_k]
            eval_candidates = [available[i] for i in top_gain_idx]
            eval_gains = gains[top_gain_idx]
        else:
            eval_candidates = available
            eval_gains = gains

        # 4. Compute marginal privacy risk Delta L_t for candidates
        marginal_risks = self.privacy_estimator.marginal_risk(
            current_qids, eval_candidates
        )

        # 5. Feasibility filtering under cumulative budget B_P
        if np.isfinite(self.privacy_budget):
            # Feasible if current leakage + prospective increase <= B_P
            feasible_mask = (cur_leakage + np.maximum(0.0, marginal_risks)) <= self.privacy_budget
            feasible_indices = np.where(feasible_mask)[0]
            if len(feasible_indices) == 0:
                raise NoFeasibleQueriesError(
                    f"No candidate satisfies privacy budget B_P={self.privacy_budget:.3f} "
                    f"(current L_hat={cur_leakage:.3f}, min prospective={cur_leakage + np.min(marginal_risks):.3f})."
                )
            feasible_candidates = [eval_candidates[i] for i in feasible_indices]
            feasible_gains = eval_gains[feasible_indices]
            feasible_risks = marginal_risks[feasible_indices]
        else:
            feasible_candidates = eval_candidates
            feasible_gains = eval_gains
            feasible_risks = marginal_risks

        self._last_feasible_count = len(feasible_candidates)
        remaining_budget = self.privacy_budget - cur_leakage if np.isfinite(self.privacy_budget) else float("inf")

        # 6. Score feasible candidates
        scores = self.scorer.score(
            feasible_gains,
            feasible_risks,
            step=len(transcript),
            remaining_privacy_budget=remaining_budget,
        )
        self._last_scores = scores

        best_pos = int(np.argmax(scores))
        return feasible_candidates[best_pos]
