"""Sequential compliance evidence model calibrated strictly on surrogate reference banks.

Sign convention (frozen across PRISM-U):
- Positive LLR_t > 0 supports H_C (compliant / properly unlearned).
- Negative LLR_t < 0 supports H_N (non-compliant / incompletely unlearned).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

EPS = 1e-6


def to_clr(probs: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Centered Log-Ratio (CLR) transform mapping simplex probabilities to R^K."""
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    log_p = np.log(p)
    return log_p - log_p.mean(axis=-1, keepdims=True)


def _logsumexp(a: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


@dataclass
class EvidenceState:
    """Current sequential evidence state after t observed responses."""

    step: int
    llr: float
    p_compliant: float
    p_noncompliant: float
    log_post_hc: np.ndarray
    log_post_hn: np.ndarray


class SurrogateEvidenceModel:
    """Transcript-adaptive reference-ensemble evidence model.

    Maintains posterior weights over the surrogate H_C and H_N reference models
    given the observed transcript tau_t = {(x_1, r_1), ..., (x_t, r_t)}.
    """

    def __init__(
        self,
        *,
        hc_probs: np.ndarray,
        hn_probs: np.ndarray,
        bandwidth: float = 1.0,
        prior_hc: float = 0.5,
        access_mode: str = "probabilities",
        use_clr: bool = True,
    ) -> None:
        if access_mode not in ("probabilities", "top1"):
            raise ValueError(f"Unsupported access_mode '{access_mode}' for SurrogateEvidenceModel")
        if not (0.0 < prior_hc < 1.0):
            raise ValueError("prior_hc must be in (0, 1)")
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be > 0")

        self.access_mode = str(access_mode)
        self.bandwidth = float(bandwidth)
        self.prior_hc = float(prior_hc)
        self.use_clr = bool(use_clr)

        # Store probability arrays (read-only copies)
        self.hc_probs = np.array(hc_probs, dtype=np.float64, copy=True)
        self.hn_probs = np.array(hn_probs, dtype=np.float64, copy=True)
        self.hc_probs.setflags(write=False)
        self.hn_probs.setflags(write=False)

        self.n_hc, self.num_queries, self.num_classes = self.hc_probs.shape
        self.n_hn = self.hn_probs.shape[0]

        if self.access_mode == "probabilities":
            self.hc_repr = to_clr(self.hc_probs) if self.use_clr else self.hc_probs
            self.hn_repr = to_clr(self.hn_probs) if self.use_clr else self.hn_probs
        else:
            # Categorical distribution smoothed slightly for top-1 likelihoods
            self.hc_repr = np.clip(self.hc_probs, 1e-4, 1.0)
            self.hn_repr = np.clip(self.hn_probs, 1e-4, 1.0)

        self.hc_repr.setflags(write=False)
        self.hn_repr.setflags(write=False)

    @classmethod
    def from_surrogate_cache(
        cls,
        cache_path,
        *,
        bandwidth: Optional[float] = None,
        prior_hc: float = 0.5,
        access_mode: str = "probabilities",
        use_clr: bool = True,
    ) -> "SurrogateEvidenceModel":
        """Build evidence model strictly from surrogate_responses.npz (excluding EXPERIMENTAL)."""
        data = np.load(cache_path, allow_pickle=False)
        probs = data["probabilities"].astype(np.float64)
        hypotheses = data["hypotheses"].astype(str)
        seeds = data["seeds"].astype(np.int64)

        hc_mask = hypotheses == "HC"
        hn_mask = hypotheses == "HN"
        if not hc_mask.any() or not hn_mask.any():
            raise RuntimeError("Surrogate cache must contain both HC and HN models.")

        hc_probs = probs[hc_mask]
        hn_probs = probs[hn_mask]

        if bandwidth is None:
            bandwidth = cls.calibrate_bandwidth_loso(
                probs=probs[hc_mask | hn_mask],
                hypotheses=hypotheses[hc_mask | hn_mask],
                seeds=seeds[hc_mask | hn_mask],
                access_mode=access_mode,
                use_clr=use_clr,
            )

        return cls(
            hc_probs=hc_probs,
            hn_probs=hn_probs,
            bandwidth=float(bandwidth),
            prior_hc=prior_hc,
            access_mode=access_mode,
            use_clr=use_clr,
        )

    @staticmethod
    def calibrate_bandwidth_loso(
        *,
        probs: np.ndarray,
        hypotheses: np.ndarray,
        seeds: np.ndarray,
        access_mode: str = "probabilities",
        use_clr: bool = True,
        candidate_bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0),
        eval_k: int = 40,
    ) -> float:
        """Select likelihood bandwidth via Leave-One-Seed-Out CV purely within surrogate seeds.

        Minimizes cross-entropy (log loss) over the top-k JS-ranked queries within each
        LOSO fold, ensuring zero leakage into validation or final target sets.
        """
        if access_mode == "top1":
            return 1.0

        unique_seeds = np.unique(seeds)
        if len(unique_seeds) < 3:
            return 4.0

        best_bw = candidate_bandwidths[0]
        best_nll = float("inf")

        for bw in candidate_bandwidths:
            losses = []
            for held_seed in unique_seeds:
                train_mask = seeds != held_seed
                test_mask = seeds == held_seed

                tr_probs = probs[train_mask]
                tr_hyp = hypotheses[train_mask]
                te_probs = probs[test_mask]
                te_hyp = hypotheses[test_mask]

                tr_hc = tr_probs[tr_hyp == "HC"]
                tr_hn = tr_probs[tr_hyp == "HN"]

                # Compute JS ranking using training seeds only
                mc = np.clip(tr_hc.mean(axis=0), 1e-12, 1.0)
                mn = np.clip(tr_hn.mean(axis=0), 1e-12, 1.0)
                mm = 0.5 * (mc + mn)
                js = 0.5 * np.sum(mc * np.log(mc / mm), axis=1) + 0.5 * np.sum(mn * np.log(mn / mm), axis=1)
                top_q = np.argsort(-js)[: min(eval_k, len(js))]

                model = SurrogateEvidenceModel(
                    hc_probs=tr_hc,
                    hn_probs=tr_hn,
                    bandwidth=bw,
                    access_mode=access_mode,
                    use_clr=use_clr,
                )
                for i in range(len(te_probs)):
                    state = model.initial_state()
                    for q in top_q:
                        state = model.update(state, int(q), te_probs[i, q])
                    y = 1.0 if te_hyp[i] == "HC" else 0.0
                    p = float(np.clip(state.p_compliant, 1e-9, 1.0 - 1e-9))
                    losses.append(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

            mean_nll = float(np.mean(losses))
            if mean_nll < best_nll:
                best_nll = mean_nll
                best_bw = bw

        return float(best_bw)

    def initial_state(self) -> EvidenceState:
        """Return prior evidence state before any queries (tau_0 = empty)."""
        log_hc = np.full(self.n_hc, np.log(self.prior_hc / self.n_hc), dtype=np.float64)
        log_hn = np.full(self.n_hn, np.log((1.0 - self.prior_hc) / self.n_hn), dtype=np.float64)
        llr = float(np.log(self.prior_hc / (1.0 - self.prior_hc)))
        return EvidenceState(
            step=0,
            llr=llr,
            p_compliant=self.prior_hc,
            p_noncompliant=1.0 - self.prior_hc,
            log_post_hc=log_hc,
            log_post_hn=log_hn,
        )

    def _encode_response(self, response: np.ndarray | int) -> np.ndarray | int:
        if self.access_mode == "top1":
            return int(np.asarray(response).item())
        vec = np.asarray(response, dtype=np.float64)
        return to_clr(vec) if self.use_clr else vec

    def step_log_likelihoods(self, query_id: int, response: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
        """Compute log P(r | x_q, m) up to a constant across all H_C and H_N reference models."""
        if self.access_mode == "top1":
            pred = int(np.asarray(response).item())
            ll_hc = np.log(self.hc_repr[:, query_id, pred]) / self.bandwidth
            ll_hn = np.log(self.hn_repr[:, query_id, pred]) / self.bandwidth
            return ll_hc, ll_hn

        obs = self._encode_response(response)
        diff_hc = self.hc_repr[:, query_id, :] - obs[None, :]
        diff_hn = self.hn_repr[:, query_id, :] - obs[None, :]
        scale = 2.0 * (self.bandwidth ** 2) * self.num_classes
        ll_hc = -np.sum(diff_hc ** 2, axis=1) / scale
        ll_hn = -np.sum(diff_hn ** 2, axis=1) / scale
        return ll_hc, ll_hn

    def update(self, state: EvidenceState, query_id: int, response: np.ndarray | int) -> EvidenceState:
        """Update posterior model weights and LLR after observing (query_id, response)."""
        ll_hc, ll_hn = self.step_log_likelihoods(query_id, response)
        unnorm_hc = state.log_post_hc + ll_hc
        unnorm_hn = state.log_post_hn + ll_hn

        log_z_hc = float(_logsumexp(unnorm_hc, axis=0))
        log_z_hn = float(_logsumexp(unnorm_hn, axis=0))
        log_z_total = float(_logsumexp(np.array([log_z_hc, log_z_hn]), axis=0))

        log_post_hc = unnorm_hc - log_z_total
        log_post_hn = unnorm_hn - log_z_total
        llr = float(log_z_hc - log_z_hn)
        p_hc = float(np.exp(log_z_hc - log_z_total))
        p_hn = float(np.exp(log_z_hn - log_z_total))

        return EvidenceState(
            step=state.step + 1,
            llr=llr,
            p_compliant=p_hc,
            p_noncompliant=p_hn,
            log_post_hc=log_post_hc,
            log_post_hn=log_post_hn,
        )

    def score_transcript(self, query_ids: list[int], responses: np.ndarray) -> EvidenceState:
        """Compute sequential evidence state from scratch given a full transcript."""
        state = self.initial_state()
        for qid, resp in zip(query_ids, responses):
            state = self.update(state, int(qid), resp)
        return state

