"""Generate publication-grade heatmaps and summary report for 2D budget sweep."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from evaluation.plots import plot_feasibility_heatmap


def main():
    repo_root = Path(__file__).resolve().parents[1]
    matrix_path = repo_root / "results" / "budget_sweep_matrix.json"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    with open(matrix_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grid = data["grid"]
    bq_values = data["bq_values"]
    bp_raw = data["bp_values"]
    bp_values = [float(p) if p != "inf" else float("inf") for p in bp_raw]

    n_bp = len(bp_values)
    n_bq = len(bq_values)

    coverage_mat = np.zeros((n_bp, n_bq), dtype=np.float64)
    auroc_mat = np.zeros((n_bp, n_bq), dtype=np.float64)
    acc_mat = np.zeros((n_bp, n_bq), dtype=np.float64)
    queries_mat = np.zeros((n_bp, n_bq), dtype=np.float64)
    leakage_mat = np.zeros((n_bp, n_bq), dtype=np.float64)

    # Index map
    bp_idx_map = {p: i for i, p in enumerate(bp_values)}
    bq_idx_map = {b: j for j, b in enumerate(bq_values)}

    for cell in grid:
        p_val = float(cell["privacy_budget"]) if cell["privacy_budget"] != "inf" else float("inf")
        b_val = int(cell["query_budget"])
        i = bp_idx_map[p_val]
        j = bq_idx_map[b_val]

        coverage_mat[i, j] = cell["coverage"]
        auroc_mat[i, j] = cell["auroc"]
        acc_mat[i, j] = cell["conditional_accuracy"] if not np.isnan(cell["conditional_accuracy"]) else 0.0
        queries_mat[i, j] = cell["mean_queries"]
        leakage_mat[i, j] = cell["mean_leakage"]

    results_dir = repo_root / "results"

    # 1. Coverage heatmap
    plot_feasibility_heatmap(
        coverage_mat,
        bq_values=bq_values,
        bp_values=bp_values,
        output_path=results_dir / "feasibility_coverage_heatmap.png",
        metric_label="Coverage (Fraction Decided)",
        title=r"PRISM-U Feasibility Phase Diagram: Coverage $\kappa$",
    )

    # 2. AUROC heatmap
    plot_feasibility_heatmap(
        auroc_mat,
        bq_values=bq_values,
        bp_values=bp_values,
        output_path=results_dir / "feasibility_auroc_heatmap.png",
        metric_label="Audit AUROC",
        title="PRISM-U Feasibility Phase Diagram: Audit AUROC",
    )

    # 3. Conditional Accuracy heatmap
    plot_feasibility_heatmap(
        acc_mat,
        bq_values=bq_values,
        bp_values=bp_values,
        output_path=results_dir / "feasibility_accuracy_heatmap.png",
        metric_label="Conditional Accuracy",
        title="PRISM-U Feasibility Phase Diagram: Conditional Accuracy",
    )

    # 4. Generate Markdown summary table
    md_lines = [
        "# PRISM-U 2D Budget Sweep Matrix Evaluation",
        "",
        "Evaluation over the 24-model Validation Bank across varying Query Budgets ($B_Q$) and Privacy Budgets ($B_P$).",
        "",
        "| Privacy Budget ($B_P$) | Query Budget ($B_Q$) | Coverage | Inconclusive Rate | False-PASS | Cond. Accuracy | Audit AUROC | Mean Queries $\\mathbb{E}[T]$ | Mean Leakage ($\\hat{L}$) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for cell in grid:
        bp_str = f"{cell['privacy_budget']:.2f}" if cell["privacy_budget"] != "inf" else "$\\infty$"
        bq_str = str(cell["query_budget"])
        cov_str = f"{cell['coverage'] * 100:.1f}%"
        inc_str = f"{cell['inconclusive_rate'] * 100:.1f}%"
        fp_str = f"{cell['false_pass_rate'] * 100:.1f}%"
        acc_str = f"{cell['conditional_accuracy'] * 100:.1f}%" if not np.isnan(cell["conditional_accuracy"]) else "N/A"
        auroc_str = f"{cell['auroc']:.4f}"
        queries_str = f"{cell['mean_queries']:.1f}"
        leakage_str = f"{cell['mean_leakage']:.4f}"

        md_lines.append(
            f"| {bp_str} | {bq_str} | {cov_str} | {inc_str} | {fp_str} | {acc_str} | {auroc_str} | {queries_str} | {leakage_str} |"
        )

    md_lines.append("")
    summary_path = results_dir / "budget_sweep_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Generated heatmaps in {results_dir} and report {summary_path}")


if __name__ == "__main__":
    main()
