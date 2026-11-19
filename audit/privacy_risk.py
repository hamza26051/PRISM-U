"""Transcript-dependent marginal retained-set privacy risk estimator.

Estimates the marginal retained-set membership leakage Delta L_t(x | tau_t)
*before* issuing a candidate query x, using shadow surrogate experiments.

Core abstractions:
    - L_hat(tau_t): Cumulative estimated leakage of the current transcript, measured
      as the best MIA AUROC achievable from the query-response history.
    - Delta L_t(x | tau_t): Marginal prospective leakage increase from adding query x:
        Delta L_t(x | tau_t) = E_r [ L_hat(tau_t ∪ {(x, r)}) ] - L_hat(tau_t)
    - R(x): Static per-query risk baseline (transcript-independent, for ablation).

Design constraints:
    - Operates entirely on pre-cached surrogate responses (no GPU, no model access).
    - Compatible with the PRISM-U acquisition policy (Phase 12) interface.
    - Provides both adaptive (transcript-dependent) and static risk estimation.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from audit.evidence import SurrogateEvidenceModel, to_clr

EPS = 1e-12


class StaticQueryRisk:
    """Static per-query retained-set risk baseline R(x).

    Computes a transcript-independent risk score for each query in the auxiliary
    pool based on how well the single-query response distinguishes HC (trained on
    D_r only) from HN (trained on D_r ∪ D_u) surrogate models.

    R(x) = MIA AUROC of a threshold attack using the response at query x to
    separate HC from HN surrogates.
    """

    def __init__(
        self,
        *,
        hc_probs: np.ndarray,
        hn_probs: np.ndarray,
    ) -> None:
        """
        Args:
            hc_probs: HC surrogate responses, shape [n_hc, Q, K].
            hn_probs: HN surrogate responses, shape [n_hn, Q, K].
        """
        self._hc_probs = np.asarray(hc_probs, dtype=np.float64)
        self._hn_probs = np.asarray(hn_probs, dtype=np.float64)
        self._n_hc = self._hc_probs.shape[0]
        self._n_hn = self._hn_probs.shape[0]
        self._num_queries = self._hc_probs.shape[1]

        # Pre-compute static risk scores for all queries
        self._risk_scores = self._compute_all_risks()
        self._risk_scores.setflags(write=False)

    def _compute_all_risks(self) -> np.ndarray:
        """Compute R(x) for every query x in the pool."""
        risks = np.empty(self._num_queries, dtype=np.float64)
        labels = np.concatenate([
            np.ones(self._n_hc, dtype=np.int64),
            np.zeros(self._n_hn, dtype=np.int64),
        ])

        for q in range(self._num_queries):
            # Use entropy of the response as a simple membership signal
            hc_resp = self._hc_probs[:, q, :]  # [n_hc, K]
            hn_resp = self._hn_probs[:, q, :]  # [n_hn, K]
            all_resp = np.vstack([hc_resp, hn_resp])  # [n_hc + n_hn, K]

            # Confidence-based score (max probability)
            conf_scores = np.max(all_resp, axis=1)
            try:
                auroc = float(roc_auc_score(labels, conf_scores))
                # Flip if below 0.5 (AUROC symmetry)
                risks[q] = max(auroc, 1.0 - auroc)
            except ValueError:
                risks[q] = 0.5  # Degenerate case

        return risks

    @classmethod
    def from_evidence_model(cls, evidence_model: SurrogateEvidenceModel) -> "StaticQueryRisk":
        """Construct from an existing SurrogateEvidenceModel."""
        return cls(
            hc_probs=evidence_model.hc_probs,
            hn_probs=evidence_model.hn_probs,
        )

    def score_candidates(self, candidate_ids: Sequence[int]) -> np.ndarray:
        """Return static risk R(x) for each candidate query_id.

        Returns:
            Array of shape [len(candidate_ids)] with risk scores in [0.5, 1.0].
            Score of 0.5 → no leakage; 1.0 → perfect membership discrimination.
        """
        idx = np.asarray(candidate_ids, dtype=np.int64)
        return self._risk_scores[idx].copy()

    @property
    def all_risks(self) -> np.ndarray:
        """Return read-only risk scores for all queries."""
        return self._risk_scores


class TranscriptPrivacyEstimator:
    """Transcript-dependent marginal privacy risk estimator.

    Maintains a running estimate of cumulative leakage L_hat(tau_t) and computes
    the marginal leakage increase Delta L_t(x | tau_t) for candidate queries.

    Implementation uses shadow surrogate experiments:
    - For each candidate x, simulate appending its response from each surrogate model.
    - Measure how much the extended transcript improves HC-vs-HN discrimination.
    - Average over response simulations to get expected marginal leakage.

    The cumulative leakage L_hat(tau_t) is defined as the best achievable MIA AUROC
    from the transcript using a simple feature-based classifier.
    """

    def __init__(
        self,
        *,
        hc_probs: np.ndarray,
        hn_probs: np.ndarray,
        static_risk: Optional[StaticQueryRisk] = None,
        fast_mode: bool = True,
    ) -> None:
        """
        Args:
            hc_probs: HC surrogate responses, shape [n_hc, Q, K].
            hn_probs: HN surrogate responses, shape [n_hn, Q, K].
            static_risk: Pre-computed static risk (created if not provided).
            fast_mode: If True, use confidence-based leakage proxy for speed.
                       If False, use full classifier-based estimation.
        """
        self._hc_probs = np.asarray(hc_probs, dtype=np.float64)
        self._hn_probs = np.asarray(hn_probs, dtype=np.float64)
        self._n_hc = self._hc_probs.shape[0]
        self._n_hn = self._hn_probs.shape[0]
        self._num_queries = self._hc_probs.shape[1]
        self._num_classes = self._hc_probs.shape[2]
        self._fast_mode = bool(fast_mode)

        self._static_risk = static_risk or StaticQueryRisk(
            hc_probs=self._hc_probs,
            hn_probs=self._hn_probs,
        )

        # Membership labels for surrogate models
        self._labels = np.concatenate([
            np.ones(self._n_hc, dtype=np.int64),
            np.zeros(self._n_hn, dtype=np.int64),
        ])

        # Precompute per-query individual leakage scores for fast_mode O(1) evaluation
        self._query_leakages = self._precompute_query_leakages()
        self._query_leakages.setflags(write=False)

    def _precompute_query_leakages(self) -> np.ndarray:
        """Precompute maximum single-feature AUROC for every query in pool Q."""
        all_hc = self._hc_probs  # [n_hc, Q, K]
        all_hn = self._hn_probs  # [n_hn, Q, K]
        all_resp = np.concatenate([all_hc, all_hn], axis=0)  # [M, Q, K]

        # 1. Max confidence: [M, Q]
        conf = np.max(all_resp, axis=2)
        # 2. Entropy: [M, Q]
        p_clip = np.clip(all_resp, EPS, 1.0)
        entropy = -np.sum(p_clip * np.log(p_clip), axis=2)
        # 3. Margin: [M, Q]
        sorted_p = np.sort(all_resp, axis=2)[:, :, ::-1]
        margin = sorted_p[:, :, 0] - sorted_p[:, :, 1]

        leakages = np.empty(self._num_queries, dtype=np.float64)
        pos_mask = self._labels == 1
        neg_mask = self._labels == 0
        n_pos = float(np.sum(pos_mask))
        n_neg = float(np.sum(neg_mask))
        denom = n_pos * n_neg

        for q in range(self._num_queries):
            best_q = 0.5
            for feat in (conf[:, q], entropy[:, q], margin[:, q]):
                pos = feat[pos_mask]
                neg = feat[neg_mask]
                diff = pos[:, None] - neg[None, :]
                u = (np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / denom
                best_q = max(best_q, u, 1.0 - u)
            leakages[q] = best_q

        return leakages

    @classmethod
    def from_evidence_model(
        cls,
        evidence_model: SurrogateEvidenceModel,
        *,
        fast_mode: bool = True,
    ) -> "TranscriptPrivacyEstimator":
        """Construct from an existing SurrogateEvidenceModel."""
        static_risk = StaticQueryRisk(
            hc_probs=evidence_model.hc_probs,
            hn_probs=evidence_model.hn_probs,
        )
        return cls(
            hc_probs=evidence_model.hc_probs,
            hn_probs=evidence_model.hn_probs,
            static_risk=static_risk,
            fast_mode=fast_mode,
        )

    def _build_transcript_features(self, query_ids: Sequence[int]) -> np.ndarray:
        """Build feature matrix from surrogate responses at the given query indices.

        Returns:
            Feature matrix of shape [n_hc + n_hn, D] where D is the feature dimension.
        """
        qids = list(query_ids)
        if not qids:
            return np.empty((self._n_hc + self._n_hn, 0), dtype=np.float64)

        # Extract responses at queried indices for all surrogates
        hc_resp = self._hc_probs[:, qids, :]  # [n_hc, T, K]
        hn_resp = self._hn_probs[:, qids, :]  # [n_hn, T, K]
        all_resp = np.concatenate([hc_resp, hn_resp], axis=0)  # [M, T, K]

        if self._fast_mode:
            # Fast feature extraction: per-query statistics
            # For each model, compute [max_conf, entropy, margin] per query position
            features_list = []
            for t in range(len(qids)):
                resp_t = all_resp[:, t, :]  # [M, K]
                max_conf = np.max(resp_t, axis=1, keepdims=True)  # [M, 1]
                p_clip = np.clip(resp_t, EPS, 1.0)
                entropy = -np.sum(p_clip * np.log(p_clip), axis=1, keepdims=True)  # [M, 1]
                sorted_p = np.sort(resp_t, axis=1)[:, ::-1]
                margin = (sorted_p[:, 0:1] - sorted_p[:, 1:2])  # [M, 1]
                features_list.extend([max_conf, entropy, margin])
            features = np.hstack(features_list)  # [M, 3*T]
        else:
            # Full features: flatten all CLR-transformed responses
            clr_all = to_clr(all_resp.reshape(-1, self._num_classes))
            clr_all = clr_all.reshape(all_resp.shape[0], -1)  # [M, T*K]
            # Also append aggregated statistics
            p_clip = np.clip(all_resp, EPS, 1.0)
            mean_entropy = -np.sum(p_clip * np.log(p_clip), axis=-1).mean(axis=1, keepdims=True)
            mean_conf = np.max(all_resp, axis=-1).mean(axis=1, keepdims=True)
            features = np.hstack([clr_all, mean_entropy, mean_conf])  # [M, T*K + 2]

        return features

    def _estimate_leakage_from_features(self, features: np.ndarray) -> float:
        """Estimate leakage (MIA AUROC) from a feature matrix over surrogate models.

        Uses the best single-feature threshold attack (highest AUROC over features)
        in fast_mode, or logistic regression in full mode.

        Returns:
            Estimated leakage L_hat in [0.5, 1.0].
        """
        if features.shape[1] == 0:
            return 0.5

        if self._fast_mode:
            # Best single-feature AUROC
            best_auroc = 0.5
            for col in range(features.shape[1]):
                scores = features[:, col]
                try:
                    auroc = float(roc_auc_score(self._labels, scores))
                    auroc = max(auroc, 1.0 - auroc)  # Handle sign flip
                    best_auroc = max(best_auroc, auroc)
                except ValueError:
                    pass
            return best_auroc
        else:
            # Logistic regression classifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_predict

            if features.shape[0] < 6 or len(np.unique(self._labels)) < 2:
                return 0.5

            try:
                clf = LogisticRegression(
                    C=1.0, max_iter=500, random_state=42, solver="lbfgs"
                )
                # Leave-one-out cross-validation scores
                n_folds = min(5, features.shape[0])
                scores = cross_val_predict(
                    clf, features, self._labels,
                    cv=n_folds, method="predict_proba",
                )[:, 1]
                auroc = float(roc_auc_score(self._labels, scores))
                return max(auroc, 1.0 - auroc)
            except Exception:
                return 0.5

    def cumulative_leakage(self, query_ids: Sequence[int]) -> float:
        """Estimate cumulative leakage L_hat(tau_t) of the current transcript.

        Args:
            query_ids: Sequence of query indices issued so far.

        Returns:
            Estimated leakage in [0.5, 1.0]. 0.5 = no leakage, 1.0 = full leakage.
        """
        if not query_ids:
            return 0.5

        if self._fast_mode:
            q_idx = [int(q) for q in query_ids]
            return max(0.5, float(np.max(self._query_leakages[q_idx])))

        features = self._build_transcript_features(query_ids)
        return self._estimate_leakage_from_features(features)

    def marginal_risk(
        self,
        query_ids: Sequence[int],
        candidate_ids: Sequence[int],
    ) -> np.ndarray:
        """Estimate marginal risk Delta L_t(x | tau_t) for each candidate query.

        Delta L_t(x | tau_t) = E_r [ L_hat(tau_t ∪ {(x, r)}) ] - L_hat(tau_t)

        Since we evaluate over surrogate models (not individual responses), the
        expectation over r is implicit: each surrogate model provides one realization
        of r, and the leakage estimator sees all of them simultaneously.

        Args:
            query_ids: Sequence of query indices issued so far in the transcript.
            candidate_ids: Sequence of candidate query indices to evaluate.

        Returns:
            Array of shape [len(candidate_ids)] with marginal risk estimates.
            Values are in [-0.5, 0.5]; positive = increases leakage.
        """
        candidates = [int(q) for q in candidate_ids]
        if not candidates:
            return np.empty((0,), dtype=np.float64)

        current_qids = list(query_ids)
        current_leakage = self.cumulative_leakage(current_qids)

        if self._fast_mode:
            cand_arr = np.asarray(candidates, dtype=np.int64)
            cand_leak = self._query_leakages[cand_arr]
            extended = np.maximum(current_leakage, cand_leak)
            return extended - current_leakage

        marginal_risks = np.empty(len(candidates), dtype=np.float64)
        for i, cand in enumerate(candidates):
            extended_qids = current_qids + [int(cand)]
            extended_leakage = self.cumulative_leakage(extended_qids)
            marginal_risks[i] = extended_leakage - current_leakage

        return marginal_risks

    def marginal_risk_batched(
        self,
        query_ids: Sequence[int],
        candidate_ids: Sequence[int],
        *,
        top_k: Optional[int] = None,
        pre_filter_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batched marginal risk estimation with optional pre-filtering.

        For large candidate pools, pre-filter to top_k candidates by static risk
        or by provided scores to avoid O(|Q|) full leakage evaluations.

        Args:
            query_ids: Current transcript query indices.
            candidate_ids: Full candidate pool.
            top_k: If set, evaluate marginal risk only for top_k candidates by
                   pre_filter_scores (or static risk), setting rest to static risk.
            pre_filter_scores: Optional scores to rank candidates for pre-filtering.

        Returns:
            Array of marginal risk estimates for all candidate_ids.
        """
        candidates = list(candidate_ids)
        n = len(candidates)
        if n == 0:
            return np.empty((0,), dtype=np.float64)

        # If no pre-filtering, compute marginal risk for all
        if top_k is None or top_k >= n:
            return self.marginal_risk(query_ids, candidates)

        # Pre-filter: select top_k by static risk or provided scores
        if pre_filter_scores is not None:
            rankings = np.asarray(pre_filter_scores, dtype=np.float64)
        else:
            rankings = self._static_risk.score_candidates(candidates)

        top_indices = np.argsort(-rankings)[:top_k]

        # Compute marginal risk for top candidates
        top_candidates = [candidates[i] for i in top_indices]
        top_risks = self.marginal_risk(query_ids, top_candidates)

        # Fill result: use static risk for non-evaluated candidates
        result = rankings.copy()  # Fallback to static/provided scores
        result -= 0.5  # Normalize to delta scale
        for j, idx in enumerate(top_indices):
            result[idx] = top_risks[j]

        return result

    @property
    def static_risk(self) -> StaticQueryRisk:
        """Access the underlying static risk estimator."""
        return self._static_risk
