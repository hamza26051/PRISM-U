from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAGE2_DIR = Path(__file__).resolve().parents[1]
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from prismu_stage2.core import load_yaml, save_json
from prismu_stage2.layout import resolved_paths

EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description="Compute Stage 2 reference-bank diagnostics without using final targets.")
    p.add_argument("--config", default="stage2/configs/stage2.yaml")
    return p.parse_args()


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m), axis=-1) + 0.5 * np.sum(q * np.log(q / m), axis=-1)


def compute_table(cache_path: Path) -> tuple[pd.DataFrame, dict]:
    cache = np.load(cache_path, allow_pickle=False)
    probs = cache["probabilities"].astype(np.float64)
    hypotheses = cache["hypotheses"].astype(str)
    conditions = cache["conditions"].astype(str)
    query_indices = cache["query_indices"]

    hc = probs[hypotheses == "HC"]
    hn = probs[hypotheses == "HN"]
    if len(hc) < 2 or len(hn) < 2:
        raise RuntimeError(f"Need at least 2 HC and 2 HN models in {cache_path}")

    mean_c = hc.mean(axis=0)
    mean_n = hn.mean(axis=0)
    l1 = np.abs(mean_c - mean_n).sum(axis=1)
    js = js_divergence(mean_c, mean_n)

    var_c = hc.var(axis=0).mean(axis=1)
    var_n = hn.var(axis=0).mean(axis=1)
    between = ((mean_c - mean_n) ** 2).mean(axis=1)
    fisher = between / (var_c + var_n + EPS)

    pred_c = hc.argmax(axis=2)
    pred_n = hn.argmax(axis=2)
    # Pairwise top-class mismatch probability across HC/HN reference populations.
    disagreement = np.empty(probs.shape[1], dtype=np.float64)
    for q in range(probs.shape[1]):
        disagreement[q] = (pred_c[:, q, None] != pred_n[None, :, q]).mean()

    table = pd.DataFrame({
        "query_id": np.arange(probs.shape[1], dtype=np.int64),
        "dataset_index": query_indices.astype(np.int64),
        "js_mean": js,
        "l1_mean": l1,
        "fisher_score": fisher,
        "top1_disagreement": disagreement,
        "within_hc_variance": var_c,
        "within_hn_variance": var_n,
    })
    table["rank_js"] = table["js_mean"].rank(method="first", ascending=False).astype(int)

    summary = {
        "cache": str(cache_path),
        "models_total": int(len(probs)),
        "hc_models": int(len(hc)),
        "hn_models": int(len(hn)),
        "experimental_models_excluded": int((hypotheses == "EXPERIMENTAL").sum()),
        "queries": int(probs.shape[1]),
        "classes": int(probs.shape[2]),
        "js_mean": float(js.mean()),
        "js_median": float(np.median(js)),
        "js_max": float(js.max()),
        "l1_mean": float(l1.mean()),
        "fisher_median": float(np.median(fisher)),
        "top1_disagreement_mean": float(disagreement.mean()),
    }
    return table, summary


def main():
    args = parse_args()
    repo_root = Path.cwd().resolve()
    cfg = load_yaml(Path(args.config))
    paths = resolved_paths(cfg, repo_root)
    artifact_root = paths["artifact_root"]
    response_root = artifact_root / "responses"
    diag_root = artifact_root / "diagnostics"
    diag_root.mkdir(parents=True, exist_ok=True)

    # Surrogate statistics are the ONLY statistics allowed to define the query ranking.
    surrogate_table, surrogate_summary = compute_table(response_root / "surrogate_responses.npz")
    surrogate_table.to_csv(diag_root / "query_separability.csv", index=False)
    save_json(surrogate_summary, diag_root / "surrogate_diagnostics.json")

    validation_table, validation_summary = compute_table(response_root / "validation_responses.npz")
    validation_table.to_csv(diag_root / "validation_query_separability.csv", index=False)

    ranked = surrogate_table.sort_values("js_mean", ascending=False).reset_index(drop=True)
    validation_by_id = validation_table.set_index("query_id")
    generalization = []
    for k in [10, 25, 50, 100, 250, 500, 1000]:
        k = min(k, len(ranked))
        ids = ranked.iloc[:k]["query_id"].to_numpy()
        vals = validation_by_id.loc[ids]
        generalization.append({
            "top_k_from_surrogate": int(k),
            "validation_js_mean": float(vals["js_mean"].mean()),
            "validation_l1_mean": float(vals["l1_mean"].mean()),
            "validation_top1_disagreement_mean": float(vals["top1_disagreement"].mean()),
        })
    validation_summary["surrogate_rank_generalization"] = generalization
    save_json(validation_summary, diag_root / "validation_diagnostics.json")

    plt.figure(figsize=(8, 5))
    plt.hist(surrogate_table["js_mean"].to_numpy(), bins=50)
    plt.xlabel("Jensen-Shannon divergence between HC/HN mean responses")
    plt.ylabel("Queries")
    plt.title("PRISM-U Stage 2 surrogate query separability")
    plt.tight_layout()
    plt.savefig(diag_root / "separability_histogram.png", dpi=160)
    plt.close()

    top = ranked.head(100)
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, len(top) + 1), top["js_mean"].to_numpy())
    plt.xlabel("Surrogate JS rank")
    plt.ylabel("JS divergence")
    plt.title("Top Stage 2 candidate queries")
    plt.tight_layout()
    plt.savefig(diag_root / "top100_queries.png", dpi=160)
    plt.close()

    print("Diagnostics use surrogate + validation banks only. Final seed 42 is untouched.")
    print(json.dumps(surrogate_summary, indent=2))


if __name__ == "__main__":
    main()
