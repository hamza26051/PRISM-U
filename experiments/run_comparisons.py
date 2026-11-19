"""Run all 6 standardized PRISM-U baseline policies on the Validation Bank.

Policies evaluated:
1. random: Uniformly random querying baseline.
2. fixed_topk: Static JS divergence surrogate ranking.
3. privacy_blind_active: Adaptive posterior entropy reduction G_t without privacy consideration.
4. static_privacy_aware: Static JS gain penalized by static risk G(x) - lambda * R(x).
5. adaptive_gain_static_privacy: Adaptive gain G_t(x | tau_t) penalized by static risk R(x).
6. full_prism_u: Full PRISM-U (adaptive gain, marginal risk Delta L_t, budget caps, sequential stopping).

Strict experiment hygiene:
- All policies and estimators calibrated strictly on surrogate_responses.npz (seeds 101-112).
- Evaluated on validation_responses.npz (seeds 201-204).
- Final seed 42 remains completely untouched.
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

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prism_u_mpl"))

import numpy as np

from audit.audit_gain import AdaptivePosteriorGainEstimator, PrivacyBlindActivePolicy, StaticJSGainEstimator
from audit.baseline_auditors import run_baseline_audit
from audit.evidence import SurrogateEvidenceModel
from audit.fixed_policy import FixedTopKPolicy
from audit.privacy_risk import StaticQueryRisk, TranscriptPrivacyEstimator
from audit.prism_u import (
    AdaptiveGainStaticPrivacyPolicy,
    PRISMUPolicy,
    StaticPrivacyAwarePolicy,
)
from audit.random_policy import RandomQueryPolicy
from audit.stopping import SequentialStoppingEngine, StoppingConfig
from evaluation.calibration import ThresholdCalibrator
from evaluation.metrics import bootstrap_auroc_ci, compute_auroc, compute_brier_score, compute_decision_metrics
from models.black_box_target import load_cached_target_bank
from utils.io import load_yaml, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Run PRISM-U comparative evaluations on validation bank.")
    p.add_argument("--config", default="configs/stage2.yaml")
    p.add_argument("--budgets", nargs="+", type=int, default=[5, 10, 20, 40, 80])
    p.add_argument("--privacy-budget", type=float, default=float("inf"))
    p.add_argument("--random-seeds", type=int, default=5)
    p.add_argument("--output-json", default="results/comparisons_validation.json")
    return p.parse_args()


def run_fixed_budget_eval(
    *,
    policy_name: str,
    make_policy_fn,
    val_cache: Path,
    evidence_model: SurrogateEvidenceModel,
    privacy_estimator: TranscriptPrivacyEstimator,
    budgets: list[int],
    n_runs: int = 1,
) -> dict:
    max_b = max(budgets)
    per_budget_llrs: dict[int, list[float]] = {b: [] for b in budgets}
    per_budget_probs: dict[int, list[float]] = {b: [] for b in budgets}
    per_budget_y: dict[int, list[int]] = {b: [] for b in budgets}
    per_budget_leakages: dict[int, list[float]] = {b: [] for b in budgets}
    exp_llrs: dict[int, list[float]] = {b: [] for b in budgets}
    query_sequences: list[list[int]] = []

    for run_idx in range(n_runs):
        bank = load_cached_target_bank(
            val_cache,
            query_budget=max_b,
            access_mode="probabilities",
            anonymize_ids=True,
        )
        for model_idx, item in enumerate(bank, start=1):
            gt = item["ground_truth"]
            target = item["target"]
            policy = make_policy_fn(run_idx, target.pool_size)

            res = run_baseline_audit(
                target=target,
                policy=policy,
                evidence_model=evidence_model,
                max_queries=max_b,
            )
            q_ids = [s.query_id for s in res.steps]
            if run_idx == 0:
                query_sequences.append(q_ids)

            hyp = gt["hypothesis"]
            for b in budgets:
                rec = res.state_at_budget(b)
                sub_qids = q_ids[:b]
                leak = privacy_estimator.cumulative_leakage(sub_qids)

                if hyp in ("HC", "HN"):
                    per_budget_llrs[b].append(rec.llr)
                    per_budget_probs[b].append(rec.p_compliant)
                    per_budget_y[b].append(1 if hyp == "HC" else 0)
                    per_budget_leakages[b].append(leak)
                elif hyp == "EXPERIMENTAL":
                    exp_llrs[b].append(rec.llr)

            if model_idx % 6 == 0 or model_idx == len(bank):
                print(f"  [{policy_name}] Progress: {model_idx}/{len(bank)} targets complete", flush=True)

    budget_metrics = {}
    for b in budgets:
        ci = bootstrap_auroc_ci(per_budget_llrs[b], per_budget_y[b], n_boot=500, seed=42)
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
            "mean_leakage": float(np.mean(per_budget_leakages[b])),
            "mean_llr_experimental_ga100": float(np.mean(exp_llrs[b])) if exp_llrs[b] else None,
        }

    unique_prefixes = {
        str(b): len({tuple(seq[:b]) for seq in query_sequences})
        for b in budgets
    }

    return {
        "policy": policy_name,
        "mode": "fixed_budget",
        "runs": int(n_runs),
        "budgets": budget_metrics,
        "unique_trajectories": unique_prefixes,
    }


def run_sequential_stopping_eval(
    *,
    policy_name: str,
    make_policy_fn,
    val_cache: Path,
    evidence_model: SurrogateEvidenceModel,
    privacy_estimator: TranscriptPrivacyEstimator,
    stopping_config: StoppingConfig,
) -> dict:
    engine = SequentialStoppingEngine(evidence_model=evidence_model, config=stopping_config)
    bank = load_cached_target_bank(
        val_cache,
        query_budget=stopping_config.query_budget,
        access_mode="probabilities",
        anonymize_ids=True,
    )

    decisions = []
    y_true = []
    queries_list = []
    leakages_list = []
    llrs = []
    target_results = []

    for target_idx, item in enumerate(bank, start=1):
        gt = item["ground_truth"]
        hyp = gt["hypothesis"]
        if hyp not in ("HC", "HN"):
            continue

        target = item["target"]
        policy = make_policy_fn(0, target.pool_size)
        res = engine.run_audit(
            target=target,
            policy=policy,
            privacy_estimator=privacy_estimator,
        )

        y = 1 if hyp == "HC" else 0
        decisions.append(res.verdict)
        y_true.append(y)
        queries_list.append(res.queries_used)
        leakages_list.append(res.cumulative_leakage)
        llrs.append(res.final_llr)

        target_results.append({
            "model_id": gt["model_id"],
            "hypothesis": hyp,
            "verdict": res.verdict,
            "reason": res.reason,
            "queries_used": res.queries_used,
            "final_llr": res.final_llr,
            "cumulative_leakage": res.cumulative_leakage,
        })

        if target_idx % 6 == 0 or target_idx == len(bank):
            print(f"  [{policy_name} stopping] Evaluated {target_idx}/{len(bank)} targets (latest: {res.verdict} in {res.queries_used}q)", flush=True)

    dec_metrics = compute_decision_metrics(decisions, y_true)
    auroc = compute_auroc(llrs, y_true)

    return {
        "policy": policy_name,
        "mode": "sequential_stopping",
        "decision_metrics": dec_metrics,
        "auroc": float(auroc),
        "mean_queries": float(np.mean(queries_list)),
        "median_queries": float(np.median(queries_list)),
        "mean_leakage": float(np.mean(leakages_list)),
        "max_leakage": float(np.max(leakages_list)),
        "targets": target_results,
    }


def main():
    args = parse_args()
    repo_root = Path.cwd().resolve()
    cfg = load_yaml(Path(args.config))
    artifact_root = (repo_root / cfg["paths"]["artifact_root"]).resolve()

    surrogate_cache = artifact_root / "responses" / "surrogate_responses.npz"
    val_cache = artifact_root / "responses" / "validation_responses.npz"

    print("Calibrating evidence model strictly on surrogate bank...", flush=True)
    evidence_model = SurrogateEvidenceModel.from_surrogate_cache(
        surrogate_cache,
        bandwidth=None,
    )
    print(f"Calibrated bandwidth: {evidence_model.bandwidth:.3f}", flush=True)

    privacy_estimator = TranscriptPrivacyEstimator.from_evidence_model(
        evidence_model, fast_mode=True
    )
    gain_est = AdaptivePosteriorGainEstimator(evidence_model)
    static_gain = StaticJSGainEstimator(evidence_model)
    static_risk = privacy_estimator.static_risk
    fixed_policy = FixedTopKPolicy.from_surrogate_cache(surrogate_cache, metric="js_mean")

    budgets = sorted(int(b) for b in args.budgets)
    out_data = {
        "metadata": {
            "budgets": budgets,
            "privacy_budget": args.privacy_budget,
            "bandwidth": evidence_model.bandwidth,
        },
        "fixed_budget_comparisons": {},
        "sequential_stopping_comparisons": {},
    }

    # 1. Random baseline
    print("Evaluating Policy 1/6: Random...", flush=True)
    out_data["fixed_budget_comparisons"]["random"] = run_fixed_budget_eval(
        policy_name="random",
        make_policy_fn=lambda r, ps: RandomQueryPolicy(pool_size=ps, seed=1000 + r),
        val_cache=val_cache,
        evidence_model=evidence_model,
        privacy_estimator=privacy_estimator,
        budgets=budgets,
        n_runs=args.random_seeds,
    )

    # 2. Fixed Top-K baseline
    print("Evaluating Policy 2/6: Fixed Top-K...", flush=True)
    out_data["fixed_budget_comparisons"]["fixed_topk"] = run_fixed_budget_eval(
        policy_name="fixed_topk",
        make_policy_fn=lambda r, ps: fixed_policy,
        val_cache=val_cache,
        evidence_model=evidence_model,
        privacy_estimator=privacy_estimator,
        budgets=budgets,
        n_runs=1,
    )

    # 3. Privacy-Blind Active baseline
    print("Evaluating Policy 3/6: Privacy-Blind Active...", flush=True)
    out_data["fixed_budget_comparisons"]["privacy_blind_active"] = run_fixed_budget_eval(
        policy_name="privacy_blind_active",
        make_policy_fn=lambda r, ps: PrivacyBlindActivePolicy(
            pool_size=ps, evidence_model=evidence_model, gain_estimator=gain_est
        ),
        val_cache=val_cache,
        evidence_model=evidence_model,
        privacy_estimator=privacy_estimator,
        budgets=budgets,
        n_runs=1,
    )

    # 4. Static Privacy-Aware baseline
    print("Evaluating Policy 4/6: Static Privacy-Aware...", flush=True)
    out_data["fixed_budget_comparisons"]["static_privacy_aware"] = run_fixed_budget_eval(
        policy_name="static_privacy_aware",
        make_policy_fn=lambda r, ps: StaticPrivacyAwarePolicy(
            pool_size=ps,
            evidence_model=evidence_model,
            penalty_lambda=1.0,
            static_gain=static_gain,
            static_risk=static_risk,
        ),
        val_cache=val_cache,
        evidence_model=evidence_model,
        privacy_estimator=privacy_estimator,
        budgets=budgets,
        n_runs=1,
    )

    # 5. Adaptive Gain Static Privacy baseline
    print("Evaluating Policy 5/6: Adaptive Gain Static Privacy...", flush=True)
    out_data["fixed_budget_comparisons"]["adaptive_gain_static_privacy"] = run_fixed_budget_eval(
        policy_name="adaptive_gain_static_privacy",
        make_policy_fn=lambda r, ps: AdaptiveGainStaticPrivacyPolicy(
            pool_size=ps,
            evidence_model=evidence_model,
            penalty_lambda=1.0,
            gain_estimator=gain_est,
            static_risk=static_risk,
        ),
        val_cache=val_cache,
        evidence_model=evidence_model,
        privacy_estimator=privacy_estimator,
        budgets=budgets,
        n_runs=1,
    )

    # 6. Full PRISM-U
    print("Evaluating Policy 6/6: Full PRISM-U...", flush=True)
    out_data["fixed_budget_comparisons"]["full_prism_u"] = run_fixed_budget_eval(
        policy_name="full_prism_u",
        make_policy_fn=lambda r, ps: PRISMUPolicy(
            pool_size=ps,
            evidence_model=evidence_model,
            privacy_budget=args.privacy_budget,
            privacy_estimator=privacy_estimator,
            gain_estimator=gain_est,
            penalty_lambda=1.0,
            pre_filter_k=50,
        ),
        val_cache=val_cache,
        evidence_model=evidence_model,
        privacy_estimator=privacy_estimator,
        budgets=budgets,
        n_runs=1,
    )

    # Sequential stopping evaluations
    print("\nRunning Sequential Stopping evaluations...", flush=True)
    stop_cfg = StoppingConfig(
        theta_pass=2.944,
        theta_fail=-2.944,
        query_budget=max(budgets),
        privacy_budget=args.privacy_budget,
    )

    for pol_key, name, make_pol in [
        ("privacy_blind_active", "privacy_blind_active", lambda ps: PrivacyBlindActivePolicy(ps, evidence_model, gain_est)),
        ("static_privacy_aware", "static_privacy_aware", lambda ps: StaticPrivacyAwarePolicy(ps, evidence_model, 1.0, static_gain, static_risk)),
        ("full_prism_u", "full_prism_u", lambda ps: PRISMUPolicy(ps, evidence_model, args.privacy_budget, privacy_estimator, gain_est, penalty_lambda=1.0, pre_filter_k=50)),
    ]:
        print(f"Sequential stopping for {name}...", flush=True)
        out_data["sequential_stopping_comparisons"][pol_key] = run_sequential_stopping_eval(
            policy_name=name,
            make_policy_fn=lambda r, ps, f=make_pol: f(ps),
            val_cache=val_cache,
            evidence_model=evidence_model,
            privacy_estimator=privacy_estimator,
            stopping_config=stop_cfg,
        )

    out_path = repo_root / args.output_json
    save_json(out_data, out_path)
    print(f"\nSaved comparison results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
