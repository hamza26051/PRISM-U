"""Rigorous auditing evaluation metrics and bootstrap confidence intervals.

Definitions follow Section 10 and Section 17 of PRISM_U_Antigravity_Handoff.md.
Positive class (1) = H_C (compliant / properly unlearned).
Negative class (0) = H_N (non-compliant / incompletely unlearned).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(scores: Sequence[float], y_true: Sequence[int]) -> float:
    """Compute ROC AUC distinguishing H_C (1) from H_N (0) using compliance score/LLR."""
    y_arr = np.asarray(y_true, dtype=np.int64)
    s_arr = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_arr)) < 2:
        raise ValueError("AUROC requires both H_C (1) and H_N (0) ground-truth targets.")
    return float(roc_auc_score(y_arr, s_arr))


def compute_brier_score(probs: Sequence[float], y_true: Sequence[int]) -> float:
    """Compute Brier calibration score E[(p - y)^2] for predicted compliance probabilities."""
    p_arr = np.asarray(probs, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((p_arr - y_arr) ** 2))


def compute_decision_metrics(decisions: Sequence[str], y_true: Sequence[int]) -> dict:
    """Compute coverage, INCONCLUSIVE rate, false-PASS, false-FAIL, and conditional accuracy.

    Args:
        decisions: Sequence of 'PASS', 'FAIL', or 'INCONCLUSIVE'.
        y_true: Sequence of 1 (H_C) or 0 (H_N).
    """
    dec = np.asarray(list(decisions), dtype=str)
    y = np.asarray(y_true, dtype=np.int64)
    n = len(dec)
    if n == 0:
        raise ValueError("Empty decisions sequence")

    is_pass = dec == "PASS"
    is_fail = dec == "FAIL"
    is_inc = dec == "INCONCLUSIVE"
    decided = is_pass | is_fail

    n_hc = int((y == 1).sum())
    n_hn = int((y == 0).sum())

    false_pass_rate = float(np.sum(is_pass & (y == 0)) / n_hn) if n_hn > 0 else float("nan")
    false_fail_rate = float(np.sum(is_fail & (y == 1)) / n_hc) if n_hc > 0 else float("nan")

    correct_decided = np.sum((is_pass & (y == 1)) | (is_fail & (y == 0)))
    cond_acc = float(correct_decided / decided.sum()) if decided.sum() > 0 else float("nan")

    return {
        "n_total": int(n),
        "n_hc": n_hc,
        "n_hn": n_hn,
        "pass_rate": float(is_pass.mean()),
        "fail_rate": float(is_fail.mean()),
        "inconclusive_rate": float(is_inc.mean()),
        "coverage": float(decided.mean()),
        "false_pass_rate": false_pass_rate,
        "false_fail_rate": false_fail_rate,
        "conditional_accuracy": cond_acc,
    }


def bootstrap_auroc_ci(
    scores: Sequence[float],
    y_true: Sequence[int],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Stratified bootstrap confidence interval for AUROC over targets."""
    s_arr = np.asarray(scores, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.int64)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Need both positive and negative samples for stratified bootstrap AUROC.")

    point = compute_auroc(s_arr, y_arr)
    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        p_samp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n_samp = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p_samp, n_samp])
        boot_vals[b] = roc_auc_score(y_arr[idx], s_arr[idx])

    lo = float(np.quantile(boot_vals, alpha / 2.0))
    hi = float(np.quantile(boot_vals, 1.0 - alpha / 2.0))
    return {"auroc": point, "ci_low": lo, "ci_high": hi}

