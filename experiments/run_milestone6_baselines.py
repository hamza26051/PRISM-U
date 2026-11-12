"""Run Milestone 6 baseline audits (Random, Fixed Top-k, Privacy-Blind Active) on Validation Bank.

Strictly maintains experiment hygiene:
- Evidence model and query rankings are calibrated ONLY on surrogate_responses.npz (seeds 101-112).
- Evaluated ONLY on validation_responses.npz (seeds 201-204).
- Final target bank (seed 42) remains completely untouched.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid Windows permission warning when matplotlib writes font cache inside sandbox
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prism_u_mpl"))

import matplotlib.pyplot as plt
import numpy as np

from audit.audit_gain import AdaptivePosteriorGainEstimator, PrivacyBlindActivePolicy
from audit.baseline_auditors import run_baseline_audit
from audit.evidence import SurrogateEvidenceModel
from audit.fixed_policy import FixedTopKPolicy
from audit.random_policy import RandomQueryPolicy
from evaluation.metrics import bootstrap_auroc_ci, compute_auroc, compute_brier_score
from models.black_box_target import load_cached_target_bank
from utils.io import load_yaml, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Run Milestone 6 auditing baselines on validation bank.")
    p.add_argument("--config", default="configs/stage2.yaml")
    p.add_argument("--budgets", nargs="+", type=int, default=[5, 10, 20, 40, 80, 160])
    p.add_argument("--random-seeds", type=int, default=10)
    p.add_argument("--access-mode", choices=["probabilities", "top1"], default="probabilities")
    return p.parse_args()


def _eval_policy_over_bank(
    *,
    policy_name: str,
    make_policy_fn,
    val_cache: Path,
    evidence_model: SurrogateEvidenceModel,
    budgets: list[int],
    access_mode: str,
    n_runs: int = 1,
) -> dict:
    max_b = max(budgets)
    per_budget_llrs: dict[int, list[float]] = {b: [] for b in budgets}
    per_budget_probs: dict[int, list[float]] = {b: [] for b in budgets}
    per_budget_y: dict[int, list[int]] = {b: [] for b in budgets}
    exp_llrs: dict[int, list[float]] = {b: [] for b in budgets}
    query_sequences: list[list[int]] = []

    for run_idx in range(n_runs):
        bank = load_cached_target_bank(
            val_cache,
            query_budget=max_b,
            access_mode=access_mode,
            anonymize_ids=True,
        )
        for item in bank:
            gt = item["ground_truth"]
            target = item["target"]
            policy = make_policy_fn(run_idx, target.pool_size)
            res = run_baseline_audit(
                target=target,
                policy=policy,
                evidence_model=evidence_model,
                max_queries=max_b,
            )
            if run_idx == 0:
                query_sequences.append([s.query_id for s in res.steps])

            hyp = gt["hypothesis"]
            for b in budgets:
                rec = res.state_at_budget(b)
                if hyp in ("HC", "HN"):
                    per_budget_llrs[b].append(rec.llr)
                    per_budget_probs[b].append(rec.p_compliant)
                    per_budget_y[b].append(1 if hyp == "HC" else 0)
                elif hyp == "EXPERIMENTAL":
                    exp_llrs[b].append(rec.llr)

    budget_metrics = {}
    for b in budgets:
        ci = bootstrap_auroc_ci(per_budget_llrs[b], per_budget_y[b], n_boot=1000, seed=42)
        brier = compute_brier_score(per_budget_probs[b], per_budget_y[b])
        hc_llrs = [s for s, y in zip(per_budget_llrs[b], per_budget_y[b]) if y == 1]
        hn_llrs = [s for s, y in zip(per_budget_llrs[b], per_budget_y[b]) if y == 0]
        budget_metrics[str(b)] = {
            "budget": int(b),
            "n_targets": int(len(per_budget_y[b])),
            "auroc": float(ci["auroc"]),
            "auroc_ci_low": float(ci["ci_low"]),
            "auroc_ci_high": float(ci["ci_high"]),
            "brier_score": float(brier),
            "mean_llr_hc": float(np.mean(hc_llrs)),
            "mean_llr_hn": float(np.mean(hn_llrs)),
            "mean_llr_experimental_ga100": float(np.mean(exp_llrs[b])) if exp_llrs[b] else None,
        }

    # Measure trajectory diversity across targets on run 0 (verifies response-adaptivity)
    unique_prefixes = {
        str(b): len({tuple(seq[:b]) for seq in query_sequences})
        for b in budgets
    }
    return {
        "policy": policy_name,
        "runs": int(n_runs),
        "budgets": budget_metrics,
        "unique_query_trajectories_run0": unique_prefixes,
    }


def main():
    args = parse_args()
    repo_root = Path.cwd().resolve()
    cfg = load_yaml(Path(args.config))
    artifact_root = (repo_root / cfg["paths"]["artifact_root"]).resolve()

    surrogate_cache = artifact_root / "responses" / "surrogate_responses.npz"
    val_cache = artifact_root / "responses" / "validation_responses.npz"

    print("Calibrating SurrogateEvidenceModel strictly on surrogate_responses.npz (LOSO CV)...")
    evidence_model = SurrogateEvidenceModel.from_surrogate_cache(
        surrogate_cache,
        bandwidth=None,  # Calibrates strictly via surrogate LOSO CV
        access_mode=args.access_mode,
    )
    print(f"Calibrated surrogate likelihood bandwidth (sigma) = {evidence_model.bandwidth:.3f}")

    # Precompute shared gain estimator (surrogate-only) and fixed policy
    gain_est = AdaptivePosteriorGainEstimator(evidence_model)
    fixed_policy = FixedTopKPolicy.from_surrogate_cache(surrogate_cache, metric="js_mean")

    budgets = sorted(int(b) for b in args.budgets)

    print("Evaluating Random Baseline across", args.random_seeds, "seeds...")
    random_res = _eval_policy_over_bank(
        policy_name="random",
        make_policy_fn=lambda r, ps: RandomQueryPolicy(pool_size=ps, seed=1000 + r),
        val_cache=val_cache,
        evidence_model=evidence_model,
        budgets=budgets,
        access_mode=args.access_mode,
        n_runs=args.random_seeds,
    )

    print("Evaluating Fixed Top-k Baseline...")
    fixed_res = _eval_policy_over_bank(
        policy_name="fixed_topk",
        make_policy_fn=lambda r, ps: fixed_policy,
        val_cache=val_cache,
        evidence_model=evidence_model,
        budgets=budgets,
        access_mode=args.access_mode,
        n_runs=1,
    )

    print("Evaluating Privacy-Blind Active Baseline...")
    active_res = _eval_policy_over_bank(
        policy_name="privacy_blind_active",
        make_policy_fn=lambda r, ps: PrivacyBlindActivePolicy(
            pool_size=ps, evidence_model=evidence_model, gain_estimator=gain_est
        ),
        val_cache=val_cache,
        evidence_model=evidence_model,
        budgets=budgets,
        access_mode=args.access_mode,
        n_runs=1,
    )

    report = {
        "milestone": 6,
        "evaluation_bank": "validation (seeds 201-204: 4 HC vs 16 HN; final seed 42 untouched)",
        "access_mode": args.access_mode,
        "calibrated_bandwidth_loso_surrogate": float(evidence_model.bandwidth),
        "budgets": budgets,
        "policies": {
            "random": random_res,
            "fixed_topk": fixed_res,
            "privacy_blind_active": active_res,
        },
    }

    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "milestone6_validation_baselines.json"
    save_json(report, out_json)

    # Print clean summary table
    print("\n" + "=" * 86)
    print("Milestone 6 Baseline AUROC Comparison on Validation Bank (4 HC vs 16 HN)")
    print("=" * 86)
    header = f"{'Budget B_Q':>10} | {'Random (10 seeds)':>20} | {'Fixed Top-k (JS)':>20} | {'Privacy-Blind Active':>20}"
    print(header)
    print("-" * 86)
    for b in budgets:
        r = random_res["budgets"][str(b)]
        f = fixed_res["budgets"][str(b)]
        a = active_res["budgets"][str(b)]
        print(
            f"{b:>10} | {r['auroc']:.4f} [{r['auroc_ci_low']:.3f},{r['auroc_ci_high']:.3f}] | "
            f"{f['auroc']:.4f} [{f['auroc_ci_low']:.3f},{f['auroc_ci_high']:.3f}] | "
            f"{a['auroc']:.4f} [{a['auroc_ci_low']:.3f},{a['auroc_ci_high']:.3f}]"
        )
    print("\nUnique query trajectories across 24 targets (run 0):")
    print("  Fixed Top-k:         ", fixed_res["unique_query_trajectories_run0"])
    print("  Privacy-Blind Active:", active_res["unique_query_trajectories_run0"])

    # Plot AUROC vs Query Budget
    plt.figure(figsize=(8.5, 5.2))
    for key, label, marker in [
        ("random", "Random (10 seeds)", "o"),
        ("fixed_topk", "Fixed Top-k (Surrogate JS)", "s"),
        ("privacy_blind_active", "Privacy-Blind Active (G_t)", "^"),
    ]:
        xs = budgets
        ys = [report["policies"][key]["budgets"][str(b)]["auroc"] for b in budgets]
        plt.plot(xs, ys, marker=marker, linewidth=2, label=label)

    plt.xscale("log", base=2)
    plt.xticks(budgets, [str(b) for b in budgets])
    plt.ylim(0.45, 1.03)
    plt.axhline(0.5, linestyle="--", color="gray", alpha=0.6, label="Random Chance (0.50)")
    plt.xlabel("Query Budget B_Q")
    plt.ylabel("Validation AUROC (HC vs HN)")
    plt.title(f"PRISM-U Milestone 6 Baselines (Validation Bank, {args.access_mode})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = results_dir / "milestone6_auroc_vs_budget.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"\nSaved JSON: {out_json}")
    print(f"Saved Plot: {out_png}")


if __name__ == "__main__":
    main()

