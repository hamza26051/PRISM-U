"""Component ablation suite evaluating individual PRISM-U mechanisms.

Evaluates 6 controlled component ablations on the validation bank:
- A1: No Privacy Penalty (lambda = 0, B_P = inf)
- A2: No Early Stopping (force run until B_Q)
- A3: Static Risk R(x) vs Transcript-dependent Delta L_t
- A4: Static Gain G(x) vs Adaptive Gain G_t
- A5: No Hard Feasibility Filtering (score penalty only)
- A6: Ratio Acquisition vs Additive Acquisition
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from audit.acquisition import AdditiveAcquisitionScorer, RatioAcquisitionScorer
from audit.audit_gain import AdaptivePosteriorGainEstimator, StaticJSGainEstimator
from audit.evidence import SurrogateEvidenceModel
from audit.privacy_risk import StaticQueryRisk, TranscriptPrivacyEstimator
from audit.prism_u import PRISMUPolicy, StaticPrivacyAwarePolicy
from audit.stopping import SequentialStoppingEngine, StoppingConfig
from evaluation.metrics import compute_auroc, compute_decision_metrics
from models.black_box_target import load_cached_target_bank
from utils.io import load_yaml, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Run PRISM-U component ablations.")
    p.add_argument("--config", default="configs/stage2.yaml")
    p.add_argument("--query-budget", type=int, default=40)
    p.add_argument("--privacy-budget", type=float, default=0.65)
    p.add_argument("--output-json", default="results/ablations_validation.json")
    return p.parse_args()


def run_single_ablation(
    name: str,
    make_policy_fn,
    engine: SequentialStoppingEngine,
    val_cache: Path,
    query_budget: int,
    privacy_estimator: TranscriptPrivacyEstimator,
) -> dict:
    print(f"Running Ablation: {name}...", flush=True)
    bank = load_cached_target_bank(
        val_cache,
        query_budget=query_budget,
        access_mode="probabilities",
        anonymize_ids=True,
    )
    decisions = []
    y_true = []
    queries = []
    leakages = []
    llrs = []

    for idx, item in enumerate(bank, start=1):
        gt = item["ground_truth"]
        hyp = gt["hypothesis"]
        if hyp not in ("HC", "HN"):
            continue

        target = item["target"]
        policy = make_policy_fn(target.pool_size)
        res = engine.run_audit(
            target=target,
            policy=policy,
            privacy_estimator=privacy_estimator,
        )

        y = 1 if hyp == "HC" else 0
        decisions.append(res.verdict)
        y_true.append(y)
        queries.append(res.queries_used)
        leakages.append(res.cumulative_leakage)
        llrs.append(res.final_llr)

        if idx % 6 == 0 or idx == len(bank):
            print(f"  [{name}] Evaluated {idx}/{len(bank)} targets (latest: {res.verdict} in {res.queries_used}q)", flush=True)
        llrs.append(res.final_llr)

    dec_metrics = compute_decision_metrics(decisions, y_true)
    try:
        auroc = compute_auroc(llrs, y_true)
    except ValueError:
        auroc = 0.5

    return {
        "ablation_name": name,
        "decision_metrics": dec_metrics,
        "auroc": float(auroc),
        "mean_queries": float(np.mean(queries)),
        "mean_leakage": float(np.mean(leakages)),
        "max_leakage": float(np.max(leakages)),
    }


def main():
    args = parse_args()
    repo_root = Path.cwd().resolve()
    cfg = load_yaml(Path(args.config))
    artifact_root = (repo_root / cfg["paths"]["artifact_root"]).resolve()

    surrogate_cache = artifact_root / "responses" / "surrogate_responses.npz"
    val_cache = artifact_root / "responses" / "validation_responses.npz"

    evidence_model = SurrogateEvidenceModel.from_surrogate_cache(surrogate_cache, bandwidth=None)
    gain_est = AdaptivePosteriorGainEstimator(evidence_model)
    static_gain = StaticJSGainEstimator(evidence_model)
    privacy_est = TranscriptPrivacyEstimator.from_evidence_model(evidence_model, fast_mode=True)
    static_risk = privacy_est.static_risk

    bq = args.query_budget
    bp = args.privacy_budget

    bank = load_cached_target_bank(val_cache, query_budget=bq, access_mode="probabilities", anonymize_ids=True)

    default_config = StoppingConfig(theta_pass=2.944, theta_fail=-2.944, query_budget=bq, privacy_budget=bp)
    default_engine = SequentialStoppingEngine(evidence_model=evidence_model, config=default_config)

    ablations = {}

    # Full PRISM-U Baseline for comparison
    ablations["full_prism_u"] = run_single_ablation(
        "Full PRISM-U (Default Reference)",
        lambda ps: PRISMUPolicy(ps, evidence_model, bp, privacy_est, gain_est, penalty_lambda=1.0, pre_filter_k=50),
        default_engine,
        val_cache,
        bq,
        privacy_est,
    )

    # A1: Omission of Privacy Penalty (B_P = inf, lambda = 0)
    a1_cfg = StoppingConfig(theta_pass=2.944, theta_fail=-2.944, query_budget=bq, privacy_budget=float("inf"))
    a1_engine = SequentialStoppingEngine(evidence_model=evidence_model, config=a1_cfg)
    ablations["A1_no_privacy_penalty"] = run_single_ablation(
        "A1: No Privacy Penalty (lambda=0, B_P=inf)",
        lambda ps: PRISMUPolicy(ps, evidence_model, float("inf"), privacy_est, gain_est, penalty_lambda=0.0, pre_filter_k=50),
        a1_engine,
        val_cache,
        bq,
        privacy_est,
    )

    # A2: Omission of Early Stopping (force to B_Q)
    a2_cfg = StoppingConfig(theta_pass=2.944, theta_fail=-2.944, query_budget=bq, privacy_budget=bp, force_max_budget=True)
    a2_engine = SequentialStoppingEngine(evidence_model=evidence_model, config=a2_cfg)
    ablations["A2_no_early_stopping"] = run_single_ablation(
        "A2: No Early Stopping (force to B_Q)",
        lambda ps: PRISMUPolicy(ps, evidence_model, bp, privacy_est, gain_est, penalty_lambda=1.0, pre_filter_k=50),
        a2_engine,
        val_cache,
        bq,
        privacy_est,
    )

    # A3: Static Risk R(x) instead of Transcript-Dependent Delta L_t
    ablations["A3_static_risk"] = run_single_ablation(
        "A3: Static Risk R(x) (No Transcript Conditioning)",
        lambda ps: StaticPrivacyAwarePolicy(ps, evidence_model, penalty_lambda=1.0, static_gain=static_gain, static_risk=static_risk),
        default_engine,
        val_cache,
        bq,
        privacy_est,
    )

    # A4: Static Gain G(x) with Full Marginal Risk Delta L_t
    ablations["A4_static_gain"] = run_single_ablation(
        "A4: Static Gain G(x) (No Posterior Mixture Gain)",
        lambda ps: PRISMUPolicy(ps, evidence_model, bp, privacy_est, gain_estimator=static_gain, scorer=AdditiveAcquisitionScorer(1.0), pre_filter_k=50),
        default_engine,
        val_cache,
        bq,
        privacy_est,
    )

    # A5: Omission of Hard Feasibility Filtering
    ablations["A5_no_feasibility_filter"] = run_single_ablation(
        "A5: No Feasibility Filter (Soft Penalty Only)",
        lambda ps: PRISMUPolicy(ps, evidence_model, privacy_budget=float("inf"), privacy_estimator=privacy_est, gain_estimator=gain_est, penalty_lambda=1.0, pre_filter_k=50),
        default_engine,
        val_cache,
        bq,
        privacy_est,
    )

    # A6: Ratio Acquisition vs Additive Acquisition
    ablations["A6_ratio_acquisition"] = run_single_ablation(
        "A6: Ratio Acquisition G / (Delta L + eps)",
        lambda ps: PRISMUPolicy(ps, evidence_model, bp, privacy_est, gain_est, acquisition_mode="ratio", pre_filter_k=50),
        default_engine,
        val_cache,
        bq,
        privacy_est,
    )

    out_path = repo_root / args.output_json
    save_json(ablations, out_path)
    print(f"\nSaved component ablation results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
