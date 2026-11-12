"""Fixed (non-adaptive) top-k query baseline policy ranked strictly on the surrogate bank."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from audit.transcript import AuditTranscript

EPS = 1e-12


def rank_queries_from_surrogate(cache_path: Path, metric: str = "js_mean") -> np.ndarray:
    """Rank auxiliary query_ids in descending order using surrogate HC vs HN separation."""
    data = np.load(cache_path, allow_pickle=False)
    probs = data["probabilities"].astype(np.float64)
    hypotheses = data["hypotheses"].astype(str)

    hc = probs[hypotheses == "HC"]
    hn = probs[hypotheses == "HN"]
    if len(hc) < 1 or len(hn) < 1:
        raise RuntimeError("Surrogate cache must contain both HC and HN models.")

    mean_c = np.clip(hc.mean(axis=0), EPS, 1.0)
    mean_n = np.clip(hn.mean(axis=0), EPS, 1.0)

    if metric == "js_mean":
        m = 0.5 * (mean_c + mean_n)
        scores = 0.5 * np.sum(mean_c * np.log(mean_c / m), axis=-1) + 0.5 * np.sum(mean_n * np.log(mean_n / m), axis=-1)
    elif metric == "l1_mean":
        scores = np.abs(mean_c - mean_n).sum(axis=-1)
    elif metric == "fisher":
        var_c = hc.var(axis=0).mean(axis=1)
        var_n = hn.var(axis=0).mean(axis=1)
        between = ((mean_c - mean_n) ** 2).mean(axis=1)
        scores = between / (var_c + var_n + EPS)
    else:
        raise ValueError(f"Unsupported ranking metric '{metric}'; expected 'js_mean', 'l1_mean', or 'fisher'.")

    return np.argsort(-scores, kind="stable").astype(np.int64)


class FixedTopKPolicy:
    """Non-adaptive baseline policy issuing queries in fixed surrogate-ranked order."""

    name = "fixed_topk"

    def __init__(self, ranked_query_ids: Sequence[int]) -> None:
        self.ranked_query_ids = [int(q) for q in ranked_query_ids]
        if not self.ranked_query_ids:
            raise ValueError("ranked_query_ids cannot be empty")

    @classmethod
    def from_surrogate_cache(cls, cache_path: Path, metric: str = "js_mean") -> "FixedTopKPolicy":
        """Build fixed policy directly from surrogate_responses.npz."""
        ranked = rank_queries_from_surrogate(cache_path, metric=metric)
        return cls(ranked.tolist())

    @classmethod
    def from_diagnostics_csv(cls, csv_path: Path, metric: str = "js_mean") -> "FixedTopKPolicy":
        """Build fixed policy from Stage 2 query_separability.csv."""
        df = pd.read_csv(csv_path)
        ranked = df.sort_values(metric, ascending=False, kind="mergesort")["query_id"].to_numpy(dtype=np.int64)
        return cls(ranked.tolist())

    def reset(self) -> None:
        """Stateless policy; provided for interface symmetry."""

    def select_next(self, transcript: AuditTranscript, candidate_ids: Sequence[int] | None = None) -> int:
        """Select highest-ranked unqueried candidate."""
        queried = transcript.queried_id_set
        allowed = None if candidate_ids is None else set(int(c) for c in candidate_ids)
        for qid in self.ranked_query_ids:
            if qid not in queried and (allowed is None or qid in allowed):
                return qid
        raise RuntimeError("No unqueried candidates remain in ranked list.")

