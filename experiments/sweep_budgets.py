"""Execute 2D grid sweep over query budget B_Q and privacy budget B_P."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from audit.audit_gain import AdaptivePosteriorGainEstimator
from audit.evidence import SurrogateEvidenceModel
from audit.privacy_risk import TranscriptPrivacyEstimator
from audit.prism_u import PRISMUPolicy
from audit.stopping import SequentialStoppingEngine, StoppingConfig
from evaluation.metrics import compute_auroc, compute_decision_metrics
from models.black_box_target import load_cached_target_bank
from utils.io import load_yaml, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Sweep 2D (B_P, B_Q) budget grid.")
    p.add_argument("--config", default="configs/stage2.yaml")
    p.add_argument("--query-budgets", nargs="+", type=int, default=[5, 10, 20, 40, 80])
    p.add_argument("--privacy-budgets", nargs="+", type=float, default=[0.55, 0.60, 0.70, 0.90, float("inf")])
    p.add_argument("--target-bank", choices=["validation", "final"], default="validation")
    p.add_argument("--output-json", default="results/budget_sweep_matrix.json")
    return p.parse_args()


def run_grid_sweep():
    args = parse_args()
    repo_root = Path.cwd().resolve()
    cfg = load_yaml(Path(args.config))
    artifact_root = (repo_root / cfg["paths"]["artifact_root"]).resolve()

    surrogate_cache = artifact_root / "responses" / "surrogate_responses.npz"
    target_cache = artifact_root / "responses" / f"{args.target_bank}_responses.npz"

    print("Calibrating SurrogateEvidenceModel...", flush=True)
    evidence_model = SurrogateEvidenceModel.from_surrogate_cache(surrogate_cache, bandwidth=None)
    gain_est = AdaptivePosteriorGainEstimator(evidence_model)
    privacy_est = TranscriptPrivacyEstimator.from_evidence_model(evidence_model, fast_mode=True)

    bq_list = sorted(int(b) for b in args.query_budgets)
    bp_list = [float(p) for p in args.privacy_budgets]

    results_grid = []

    for bp in bp_list:
        for bq in bq_list:
            bp_label = f"{bp:.2f}" if np.isfinite(bp) else "inf"
            print(f"Evaluating cell: B_P={bp_label}, B_Q={bq}...", flush=True)

            stop_cfg = StoppingConfig(
                theta_pass=2.944,
                theta_fail=-2.944,
                query_budget=bq,
                privacy_budget=bp,
            )
            engine = SequentialStoppingEngine(evidence_model=evidence_model, config=stop_cfg)

            bank = load_cached_target_bank(
                target_cache,
                query_budget=bq,
                access_mode="probabilities",
                anonymize_ids=True,
            )

            decisions = []
            y_true = []
            queries_used = []
            leakages = []
            llrs = []

            for item in bank:
                gt = item["ground_truth"]
                hyp = gt["hypothesis"]
                if hyp not in ("HC", "HN"):
                    continue

                target = item["target"]
                policy = PRISMUPolicy(
                    pool_size=target.pool_size,
                    evidence_model=evidence_model,
                    privacy_budget=bp,
                    privacy_estimator=privacy_est,
                    gain_estimator=gain_est,
                    penalty_lambda=1.0,
                    pre_filter_k=50,
                )

                res = engine.run_audit(
                    target=target,
                    policy=policy,
                    privacy_estimator=privacy_est,
                )

                y = 1 if hyp == "HC" else 0
                decisions.append(res.verdict)
                y_true.append(y)
                queries_used.append(res.queries_used)
                leakages.append(res.cumulative_leakage)
                llrs.append(res.final_llr)

            dec_metrics = compute_decision_metrics(decisions, y_true)
            try:
                auroc = compute_auroc(llrs, y_true)
            except ValueError:
                auroc = 0.5

            results_grid.append({
                "privacy_budget": bp if np.isfinite(bp) else "inf",
                "query_budget": bq,
                "coverage": dec_metrics["coverage"],
                "inconclusive_rate": dec_metrics["inconclusive_rate"],
                "false_pass_rate": dec_metrics["false_pass_rate"],
                "false_fail_rate": dec_metrics["false_fail_rate"],
                "conditional_accuracy": dec_metrics["conditional_accuracy"],
                "auroc": float(auroc),
                "mean_queries": float(np.mean(queries_used)),
                "mean_leakage": float(np.mean(leakages)),
            })

    out_path = repo_root / args.output_json
    save_json({"grid": results_grid, "bq_values": bq_list, "bp_values": [p if np.isfinite(p) else "inf" for p in bp_list]}, out_path)
    print(f"Saved grid sweep matrix to {out_path}", flush=True)


if __name__ == "__main__":
    run_grid_sweep()
