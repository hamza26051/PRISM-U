"""Audit gain estimators G_t(x | tau_t) and privacy-blind active baseline policy."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from audit.evidence import EvidenceState, SurrogateEvidenceModel, _logsumexp
from audit.transcript import AuditTranscript

EPS = 1e-12


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p_c = np.clip(p, EPS, 1.0 - EPS)
    return -(p_c * np.log(p_c) + (1.0 - p_c) * np.log(1.0 - p_c))


class StaticJSGainEstimator:
    """Static one-time query usefulness baseline using HC vs HN Jensen-Shannon divergence."""

    def __init__(self, evidence_model: SurrogateEvidenceModel) -> None:
        mean_c = np.clip(evidence_model.hc_probs.mean(axis=0), EPS, 1.0)
        mean_n = np.clip(evidence_model.hn_probs.mean(axis=0), EPS, 1.0)
        m = 0.5 * (mean_c + mean_n)
        js = 0.5 * np.sum(mean_c * np.log(mean_c / m), axis=-1) + 0.5 * np.sum(mean_n * np.log(mean_n / m), axis=-1)
        self._scores = np.array(js, dtype=np.float64, copy=True)
        self._scores.setflags(write=False)

    def score_candidates(self, state: EvidenceState, candidate_ids: Sequence[int]) -> np.ndarray:
        idx = np.asarray(candidate_ids, dtype=np.int64)
        return self._scores[idx].copy()


class AdaptivePosteriorGainEstimator:
    """Response-adaptive expected information gain G_t(x | tau_t).

    Computes the expected reduction in compliance hypothesis entropy H(P(H | tau_t))
    when querying candidate x under the current posterior predictive mixture over the
    surrogate H_C / H_N reference ensemble:

        G_t(x | tau_t) = H(P(H | tau_t)) - E_{r ~ P(R_x | tau_t)} [ H(P(H | tau_t, x, r)) ]
    """

    def __init__(self, evidence_model: SurrogateEvidenceModel, chunk_size: int = 1024) -> None:
        self.model = evidence_model
        self.chunk_size = int(chunk_size)
        self.n_hc = evidence_model.n_hc
        self.n_hn = evidence_model.n_hn

        if evidence_model.access_mode == "probabilities":
            repr_all = np.concatenate([evidence_model.hc_repr, evidence_model.hn_repr], axis=0)  # [M, C, K]
            norms2 = np.sum(repr_all ** 2, axis=2).T  # [C, M]
            dot = np.einsum("mck,jck->cmj", repr_all, repr_all)  # [C, M, M]
            dist2 = np.maximum(0.0, norms2[:, :, None] + norms2[:, None, :] - 2.0 * dot)
            scale = 2.0 * (evidence_model.bandwidth ** 2) * evidence_model.num_classes
            self._ll_mat = (-dist2 / scale).astype(np.float32)  # [C, M, M]
            self._probs_cqk = None
        else:
            probs_all = np.concatenate([evidence_model.hc_repr, evidence_model.hn_repr], axis=0)  # [M, C, K]
            self._probs_cqk = np.transpose(probs_all, (1, 0, 2)).astype(np.float32)  # [C, M, K]
            self._ll_mat = (np.log(self._probs_cqk) / evidence_model.bandwidth).astype(np.float32)

    def score_candidates(self, state: EvidenceState, candidate_ids: Sequence[int]) -> np.ndarray:
        """Compute G_t(x | tau_t) for every candidate query_id in candidate_ids."""
        idx = np.asarray(candidate_ids, dtype=np.int64)
        if len(idx) == 0:
            return np.empty((0,), dtype=np.float64)

        log_w = np.concatenate([state.log_post_hc, state.log_post_hn]).astype(np.float64)  # [M]
        w = np.exp(log_w)  # [M], sums to 1
        cur_ent = float(_binary_entropy(np.array([state.p_compliant]))[0])

        out = np.empty((len(idx),), dtype=np.float64)
        for start in range(0, len(idx), self.chunk_size):
            batch_idx = idx[start : start + self.chunk_size]
            ll_sub = self._ll_mat[batch_idx].astype(np.float64)  # [B, M, O]
            unnorm = ll_sub + log_w[None, :, None]  # [B, M, O]

            log_z_hc = _logsumexp(unnorm[:, : self.n_hc, :], axis=1)  # [B, O]
            log_z_hn = _logsumexp(unnorm[:, self.n_hc :, :], axis=1)  # [B, O]
            log_z_tot = np.logaddexp(log_z_hc, log_z_hn)
            p_hc_post = np.exp(log_z_hc - log_z_tot)  # [B, O]
            post_ent = _binary_entropy(p_hc_post)  # [B, O]

            if self._probs_cqk is None:
                # Simulated outcomes weighted by posterior model probabilities w[j]
                expected_ent = np.sum(post_ent * w[None, :], axis=1)
            else:
                # Discrete top-1 outcomes weighted by predictive class probabilities
                p_outcome = np.einsum("m,bmk->bk", w, self._probs_cqk[batch_idx].astype(np.float64))
                expected_ent = np.sum(post_ent * p_outcome, axis=1)

            out[start : start + len(batch_idx)] = np.maximum(0.0, cur_ent - expected_ent)

        return out


class PrivacyBlindActivePolicy:
    """Active auditor selecting argmax G_t(x | tau_t) online without privacy penalties."""

    name = "privacy_blind_active"

    def __init__(
        self,
        pool_size: int,
        evidence_model: SurrogateEvidenceModel,
        gain_estimator: AdaptivePosteriorGainEstimator | None = None,
    ) -> None:
        self.pool_size = int(pool_size)
        self.evidence_model = evidence_model
        self.gain_estimator = (
            AdaptivePosteriorGainEstimator(evidence_model)
            if gain_estimator is None
            else gain_estimator
        )
        self._last_gains: np.ndarray | None = None

    def reset(self) -> None:
        self._last_gains = None

    def select_next(
        self,
        transcript: AuditTranscript,
        state: EvidenceState | None = None,
        candidate_ids: Sequence[int] | None = None,
    ) -> int:
        """Score unqueried candidates conditioned on tau_t and return the highest-gain query_id."""
        queried = transcript.queried_id_set
        pool = range(self.pool_size) if candidate_ids is None else candidate_ids
        available = [int(q) for q in pool if int(q) not in queried]
        if not available:
            raise RuntimeError("No unqueried candidates remain in pool.")

        if state is None:
            state = self.evidence_model.score_transcript(transcript.query_ids, transcript.responses)

        gains = self.gain_estimator.score_candidates(state, available)
        self._last_gains = gains
        best_pos = int(np.argmax(gains))
        return available[best_pos]

