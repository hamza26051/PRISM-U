"""Attack evaluation framework for retained-set membership inference leakage.

Provides end-to-end evaluation pipelines operating on pre-cached model responses
(no GPU required). Measures empirical privacy leakage of audit interaction
transcripts by computing MIA AUROC and Advantage metrics.

Evaluation Protocol:
    1. Partition retained members (D_r) and non-members into attack_train / attack_eval
       subsets with disjoint sample indices.
    2. Obtain model predictions (softmax vectors) for all member and non-member samples.
    3. Fit learned attacks on attack_train; evaluate all attacks on attack_eval.
    4. Report AUROC, Advantage, and bootstrap confidence intervals.

This module is designed to be compatible with both:
    - Full-split evaluation (predictions over all D_r and non-member sets), and
    - Transcript-based evaluation (predictions only over queried audit samples).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from attacks.membership_inference import (
    ALL_ATTACKS,
    ALL_LEARNED_ATTACKS,
    ALL_METRIC_ATTACKS,
    AttackOutput,
    MembershipInferenceAttack,
)

EPS = 1e-12


@dataclass(frozen=True)
class AttackEvalResult:
    """Evaluation result for a single attack method on a single model/transcript.

    Attributes:
        attack_name: Name identifier of the attack method.
        auroc: Area Under the ROC Curve for member vs. non-member discrimination.
        advantage: MIA Advantage = 2 * (AUROC - 0.5), clamped to [0, 1].
        n_members: Number of member (retained) samples in evaluation.
        n_nonmembers: Number of non-member samples in evaluation.
        ci_low: Lower bound of 95% bootstrap CI for AUROC (optional).
        ci_high: Upper bound of 95% bootstrap CI for AUROC (optional).
    """
    attack_name: str
    auroc: float
    advantage: float
    n_members: int
    n_nonmembers: int
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "attack_name": self.attack_name,
            "auroc": self.auroc,
            "advantage": self.advantage,
            "n_members": self.n_members,
            "n_nonmembers": self.n_nonmembers,
        }
        if self.ci_low is not None:
            d["ci_low"] = self.ci_low
        if self.ci_high is not None:
            d["ci_high"] = self.ci_high
        return d


def compute_mia_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute MIA AUROC where label 1 = member, 0 = non-member.

    Higher AUROC → more leakage (attack distinguishes members from non-members).
    AUROC = 0.5 → no leakage (random guessing).
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise ValueError("MIA AUROC requires both member (1) and non-member (0) samples.")
    return float(roc_auc_score(y, s))


def compute_mia_advantage(auroc: float) -> float:
    """Compute MIA Advantage = 2 * (AUROC - 0.5), clamped to [0, 1]."""
    return float(np.clip(2.0 * (auroc - 0.5), 0.0, 1.0))


def bootstrap_mia_auroc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Stratified bootstrap confidence interval for MIA AUROC.

    Returns:
        (auroc_point, ci_low, ci_high)
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Need both members (1) and non-members (0) for bootstrap.")

    point = compute_mia_auroc(s, y)
    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        p_samp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n_samp = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p_samp, n_samp])
        try:
            boot_vals[b] = roc_auc_score(y[idx], s[idx])
        except ValueError:
            boot_vals[b] = point  # Degenerate sample

    lo = float(np.quantile(boot_vals, alpha / 2.0))
    hi = float(np.quantile(boot_vals, 1.0 - alpha / 2.0))
    return point, lo, hi


def evaluate_single_attack(
    attack: MembershipInferenceAttack,
    *,
    member_probs: np.ndarray,
    nonmember_probs: np.ndarray,
    member_labels: Optional[np.ndarray] = None,
    nonmember_labels: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    boot_seed: int = 42,
) -> AttackEvalResult:
    """Evaluate a single MIA attack on member and non-member probability vectors.

    Args:
        attack: Fitted MembershipInferenceAttack instance (learned attacks must
                already be fit on a disjoint training partition).
        member_probs: Probability vectors for retained members, shape [N_mem, K].
        nonmember_probs: Probability vectors for non-members, shape [N_non, K].
        member_labels: Ground-truth class labels for members (required for LossAttack).
        nonmember_labels: Ground-truth class labels for non-members (required for LossAttack).
        n_boot: Number of bootstrap samples for confidence interval.
        boot_seed: Random seed for bootstrap.

    Returns:
        AttackEvalResult with AUROC, Advantage, and bootstrap CI.
    """
    mem_p = np.asarray(member_probs, dtype=np.float64)
    non_p = np.asarray(nonmember_probs, dtype=np.float64)

    # Combine for scoring
    all_probs = np.vstack([mem_p, non_p])
    membership_labels = np.concatenate([
        np.ones(len(mem_p), dtype=np.int64),
        np.zeros(len(non_p), dtype=np.int64),
    ])

    # Class labels for loss-based attacks
    all_class_labels: Optional[np.ndarray] = None
    if member_labels is not None and nonmember_labels is not None:
        all_class_labels = np.concatenate([
            np.asarray(member_labels, dtype=np.int64),
            np.asarray(nonmember_labels, dtype=np.int64),
        ])

    # Run attack
    output = attack.attack(all_probs, labels=all_class_labels)
    scores = output.scores

    # Compute metrics with bootstrap CI
    auroc, ci_low, ci_high = bootstrap_mia_auroc_ci(
        scores, membership_labels,
        n_boot=n_boot, seed=boot_seed,
    )
    advantage = compute_mia_advantage(auroc)

    return AttackEvalResult(
        attack_name=attack.name,
        auroc=auroc,
        advantage=advantage,
        n_members=len(mem_p),
        n_nonmembers=len(non_p),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def evaluate_all_attacks(
    *,
    member_probs_train: np.ndarray,
    nonmember_probs_train: np.ndarray,
    member_probs_eval: np.ndarray,
    nonmember_probs_eval: np.ndarray,
    member_labels_train: Optional[np.ndarray] = None,
    nonmember_labels_train: Optional[np.ndarray] = None,
    member_labels_eval: Optional[np.ndarray] = None,
    nonmember_labels_eval: Optional[np.ndarray] = None,
    metric_attacks: Optional[Sequence[str]] = None,
    learned_attacks: Optional[Sequence[str]] = None,
    n_boot: int = 1000,
    boot_seed: int = 42,
) -> list[AttackEvalResult]:
    """Evaluate all specified MIA attacks with proper train/eval separation.

    Learned attacks are fit on (member_probs_train, nonmember_probs_train) and
    evaluated on (member_probs_eval, nonmember_probs_eval).
    Metric attacks are evaluated directly on the eval partition.

    Args:
        member_probs_train: Member probabilities for training learned attacks.
        nonmember_probs_train: Non-member probabilities for training learned attacks.
        member_probs_eval: Member probabilities for evaluation.
        nonmember_probs_eval: Non-member probabilities for evaluation.
        member_labels_*: Ground-truth class labels (needed only for LossAttack).
        nonmember_labels_*: Ground-truth class labels (needed only for LossAttack).
        metric_attacks: List of metric attack names to evaluate (default: all).
        learned_attacks: List of learned attack names to evaluate (default: all).
        n_boot: Number of bootstrap iterations.
        boot_seed: Bootstrap random seed.

    Returns:
        List of AttackEvalResult, one per attack.
    """
    if metric_attacks is None:
        metric_attacks = list(ALL_METRIC_ATTACKS.keys())
    if learned_attacks is None:
        learned_attacks = list(ALL_LEARNED_ATTACKS.keys())

    results: list[AttackEvalResult] = []

    # Metric-based attacks (no fitting needed)
    for atk_name in metric_attacks:
        if atk_name not in ALL_METRIC_ATTACKS:
            raise ValueError(f"Unknown metric attack: {atk_name}")
        atk = ALL_METRIC_ATTACKS[atk_name]()
        results.append(evaluate_single_attack(
            atk,
            member_probs=member_probs_eval,
            nonmember_probs=nonmember_probs_eval,
            member_labels=member_labels_eval,
            nonmember_labels=nonmember_labels_eval,
            n_boot=n_boot,
            boot_seed=boot_seed,
        ))

    # Learned attacks (fit on train, evaluate on eval)
    for atk_name in learned_attacks:
        if atk_name not in ALL_LEARNED_ATTACKS:
            raise ValueError(f"Unknown learned attack: {atk_name}")
        atk = ALL_LEARNED_ATTACKS[atk_name]()
        atk.fit(
            member_probs=member_probs_train,
            nonmember_probs=nonmember_probs_train,
            member_labels=member_labels_train,
            nonmember_labels=nonmember_labels_train,
        )
        results.append(evaluate_single_attack(
            atk,
            member_probs=member_probs_eval,
            nonmember_probs=nonmember_probs_eval,
            member_labels=member_labels_eval,
            nonmember_labels=nonmember_labels_eval,
            n_boot=n_boot,
            boot_seed=boot_seed,
        ))

    return results


@dataclass
class TranscriptLeakageResult:
    """Privacy leakage measured from an audit interaction transcript.

    Quantifies how much retained-set membership information is exposed by
    the specific sequence of queries and responses in a transcript.
    """
    target_id: str
    n_queries: int
    attack_results: list[AttackEvalResult]
    max_auroc: float
    max_advantage: float

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "n_queries": self.n_queries,
            "max_auroc": self.max_auroc,
            "max_advantage": self.max_advantage,
            "attacks": [r.to_dict() for r in self.attack_results],
        }


def evaluate_transcript_leakage(
    *,
    transcript_responses: np.ndarray,
    transcript_query_ids: Sequence[int],
    surrogate_probs: np.ndarray,
    surrogate_hypotheses: np.ndarray,
    target_id: str = "unknown",
    n_boot: int = 500,
    boot_seed: int = 42,
) -> TranscriptLeakageResult:
    """Evaluate retained-set membership leakage from an audit transcript.

    Measures whether the queried responses allow distinguishing HC (retrained)
    from HN (original/partial-unlearning) surrogate models — which is a proxy
    for retained-set membership leakage since HC models were trained on D_r only
    while HN models were trained on D_r ∪ D_u.

    The attack surfaces are the response vectors at the queried indices,
    treating HC responses as "member" behavior and HN as "non-member" behavior.

    Args:
        transcript_responses: Actual target responses, shape [T, K] (unused for
            surrogate-based leakage, but tracked for future calibration).
        transcript_query_ids: Indices of queries issued in the transcript.
        surrogate_probs: Full surrogate bank, shape [M, Q, K].
        surrogate_hypotheses: Hypothesis labels, shape [M].
        target_id: Identifier for logging.
        n_boot: Bootstrap iterations.
        boot_seed: Bootstrap seed.

    Returns:
        TranscriptLeakageResult with per-attack AUROC and advantage.
    """
    qids = list(transcript_query_ids)
    n_queries = len(qids)
    if n_queries == 0:
        return TranscriptLeakageResult(
            target_id=target_id,
            n_queries=0,
            attack_results=[],
            max_auroc=0.5,
            max_advantage=0.0,
        )

    hyps = np.asarray(surrogate_hypotheses, dtype=str)
    hc_mask = hyps == "HC"
    hn_mask = hyps == "HN"

    # Extract responses at queried indices for all surrogate models
    # Shape: [n_hc/n_hn, n_queries, K] → flatten to [n_models, n_queries * K]
    hc_resp = surrogate_probs[hc_mask][:, qids, :]  # [n_hc, T, K]
    hn_resp = surrogate_probs[hn_mask][:, qids, :]  # [n_hn, T, K]

    # Flatten query responses into feature vectors for each surrogate model
    hc_flat = hc_resp.reshape(hc_resp.shape[0], -1)  # [n_hc, T*K]
    hn_flat = hn_resp.reshape(hn_resp.shape[0], -1)  # [n_hn, T*K]

    # For surrogate-based leakage, we evaluate metric attacks on the
    # concatenated query response vectors. We use a simple split for
    # train/eval of learned attacks.
    n_hc = len(hc_flat)
    n_hn = len(hn_flat)

    # With small surrogate counts (12 HC, 48 HN), use leave-one-out style
    # For metric attacks, evaluate on all models directly
    # For learned attacks, use 50/50 split
    hc_half = max(1, n_hc // 2)
    hn_half = max(1, n_hn // 2)

    # Evaluate metric attacks on the response feature representation
    # We evaluate entropy and confidence on mean response per model
    # For each surrogate model, compute mean response across queried positions
    hc_mean_resp = hc_resp.mean(axis=1)  # [n_hc, K]
    hn_mean_resp = hn_resp.mean(axis=1)  # [n_hn, K]

    # Metric attacks on aggregated responses
    metric_results: list[AttackEvalResult] = []
    for atk_name in ["entropy", "confidence", "margin"]:
        atk = ALL_METRIC_ATTACKS[atk_name]()
        all_probs = np.vstack([hc_mean_resp, hn_mean_resp])
        membership = np.concatenate([np.ones(n_hc), np.zeros(n_hn)])
        output = atk.attack(all_probs)
        try:
            auroc, ci_lo, ci_hi = bootstrap_mia_auroc_ci(
                output.scores, membership.astype(np.int64),
                n_boot=n_boot, seed=boot_seed,
            )
        except ValueError:
            auroc, ci_lo, ci_hi = 0.5, 0.5, 0.5
        metric_results.append(AttackEvalResult(
            attack_name=f"transcript_{atk_name}",
            auroc=auroc,
            advantage=compute_mia_advantage(auroc),
            n_members=n_hc,
            n_nonmembers=n_hn,
            ci_low=ci_lo,
            ci_high=ci_hi,
        ))

    # Learned attack on flattened response vectors (if enough models)
    learned_results: list[AttackEvalResult] = []
    if n_hc >= 4 and n_hn >= 4:
        from attacks.membership_inference import LogisticRegressionAttack as LR

        lr_atk = LR(C=1.0, seed=boot_seed)
        # Fit on first half, eval on second half
        lr_atk._clf = None
        from sklearn.linear_model import LogisticRegression

        X_train = np.vstack([hc_flat[:hc_half], hn_flat[:hn_half]])
        y_train = np.concatenate([np.ones(hc_half), np.zeros(hn_half)])
        X_eval = np.vstack([hc_flat[hc_half:], hn_flat[hn_half:]])
        y_eval = np.concatenate([
            np.ones(n_hc - hc_half),
            np.zeros(n_hn - hn_half),
        ])

        if len(np.unique(y_eval)) >= 2 and len(X_train) >= 4:
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=boot_seed, solver="lbfgs")
            try:
                clf.fit(X_train, y_train)
                scores = clf.predict_proba(X_eval)[:, 1]
                auroc = float(roc_auc_score(y_eval.astype(int), scores))
                adv = compute_mia_advantage(auroc)
                learned_results.append(AttackEvalResult(
                    attack_name="transcript_logistic_regression",
                    auroc=auroc,
                    advantage=adv,
                    n_members=n_hc - hc_half,
                    n_nonmembers=n_hn - hn_half,
                ))
            except Exception:
                pass  # Degenerate case; skip learned attack

    all_results = metric_results + learned_results
    max_auroc = max((r.auroc for r in all_results), default=0.5)
    max_advantage = max((r.advantage for r in all_results), default=0.0)

    return TranscriptLeakageResult(
        target_id=target_id,
        n_queries=n_queries,
        attack_results=all_results,
        max_auroc=max_auroc,
        max_advantage=max_advantage,
    )


def save_evaluation_report(
    results: list[AttackEvalResult],
    path: Path,
    metadata: Optional[dict] = None,
) -> None:
    """Serialize attack evaluation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": metadata or {},
        "summary": {
            "max_auroc": max(r.auroc for r in results) if results else None,
            "max_advantage": max(r.advantage for r in results) if results else None,
            "n_attacks_evaluated": len(results),
        },
        "attacks": [r.to_dict() for r in results],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
