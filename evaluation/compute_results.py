"""Aggregates and formats PRISM-U audit evaluation results into markdown tables and summary reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table without requiring tabulate."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) if row[c] is not None else "" for c in cols) + " |")
    return "\n".join([header, separator] + rows)


def summarize_comparisons(results_json: Path) -> str:
    """Generate a clean markdown summary table from comparison results."""
    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    md_lines = ["# PRISM-U Comparative Auditing Summary\n"]

    # Fixed Budget Comparisons
    fixed_comp = data.get("fixed_budget_comparisons", {})
    if fixed_comp:
        md_lines.append("## Fixed-Budget Baseline Comparisons\n")
        rows = []
        for pol_name, p_data in fixed_comp.items():
            budgets = p_data.get("budgets", {})
            for b_str, m in budgets.items():
                rows.append({
                    "Policy": pol_name,
                    "Budget (B_Q)": int(b_str),
                    "Audit AUROC": f"{m['auroc']:.4f}",
                    "AUROC 95% CI": f"[{m['auroc_ci_low']:.4f}, {m['auroc_ci_high']:.4f}]",
                    "Brier Score": f"{m['brier_score']:.6f}",
                    "Mean Leakage (L_hat)": f"{m.get('mean_leakage', 0.5):.4f}",
                    "Unique Trajectories": p_data.get("unique_trajectories", {}).get(b_str, 1),
                })
        if rows:
            df = pd.DataFrame(rows)
            md_lines.append(_df_to_markdown(df))
            md_lines.append("\n")

    # Sequential Stopping Comparisons
    seq_comp = data.get("sequential_stopping_comparisons", {})
    if seq_comp:
        md_lines.append("## Sequential Stopping Decisions (Tri-State Verification)\n")
        rows = []
        for pol_name, s_data in seq_comp.items():
            dm = s_data.get("decision_metrics", {})
            rows.append({
                "Policy": pol_name,
                "Coverage": f"{dm.get('coverage', 0.0) * 100:.1f}%",
                "Inconclusive": f"{dm.get('inconclusive_rate', 0.0) * 100:.1f}%",
                "False-PASS": f"{dm.get('false_pass_rate', 0.0) * 100:.1f}%",
                "False-FAIL": f"{dm.get('false_fail_rate', 0.0) * 100:.1f}%",
                "Cond. Accuracy": f"{dm.get('conditional_accuracy', 0.0) * 100:.1f}%",
                "Audit AUROC": f"{s_data.get('auroc', 0.0):.4f}",
                "Mean Queries E[T]": f"{s_data.get('mean_queries', 0.0):.1f}",
                "Median Queries": f"{s_data.get('median_queries', 0.0):.1f}",
                "Mean Leakage": f"{s_data.get('mean_leakage', 0.5):.4f}",
            })
        if rows:
            df_seq = pd.DataFrame(rows)
            md_lines.append(_df_to_markdown(df_seq))
            md_lines.append("\n")

    return "\n".join(md_lines)


def summarize_ablations(ablations_json: Path) -> str:
    """Generate a clean markdown summary table from component ablation results."""
    with open(ablations_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    md_lines = ["# PRISM-U Component Ablation Suite (A1–A6)\n"]
    rows = []
    for key, item in data.items():
        dm = item.get("decision_metrics", {})
        rows.append({
            "Ablation / Setting": item.get("ablation_name", key),
            "Coverage": f"{dm.get('coverage', 0.0) * 100:.1f}%",
            "Inconclusive": f"{dm.get('inconclusive_rate', 0.0) * 100:.1f}%",
            "False-PASS": f"{dm.get('false_pass_rate', 0.0) * 100:.1f}%",
            "False-FAIL": f"{dm.get('false_fail_rate', 0.0) * 100:.1f}%",
            "Cond. Accuracy": f"{dm.get('conditional_accuracy', 0.0) * 100:.1f}%",
            "Audit AUROC": f"{item.get('auroc', 0.0):.4f}",
            "Mean Queries": f"{item.get('mean_queries', 0.0):.1f}",
            "Mean Leakage": f"{item.get('mean_leakage', 0.5):.4f}",
            "Max Leakage": f"{item.get('max_leakage', 0.5):.4f}",
        })
    if rows:
        df = pd.DataFrame(rows)
        md_lines.append(_df_to_markdown(df))
        md_lines.append("\n")

    return "\n".join(md_lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", default=None)
    p.add_argument("--output-md", default="results/comparisons_summary.md")
    p.add_argument("--ablations-json", default=None)
    p.add_argument("--ablations-md", default="results/ablations_summary.md")
    args = p.parse_args()

    if args.input_json:
        in_path = Path(args.input_json)
        if in_path.exists():
            summary = summarize_comparisons(in_path)
            out_path = Path(args.output_md)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(summary, encoding="utf-8")
            print(f"Comparison summary written to {out_path}")
            print("\n" + summary)
        else:
            print(f"Comparison input file not found: {in_path}")

    if args.ablations_json:
        abl_path = Path(args.ablations_json)
        if abl_path.exists():
            abl_summary = summarize_ablations(abl_path)
            abl_out = Path(args.ablations_md)
            abl_out.parent.mkdir(parents=True, exist_ok=True)
            abl_out.write_text(abl_summary, encoding="utf-8")
            print(f"Ablation summary written to {abl_out}")
            print("\n" + abl_summary)
        else:
            print(f"Ablation input file not found: {abl_path}")
    out_path = Path(args.output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary, encoding="utf-8")
    print(f"Summary written to {out_path}")
    print("\n" + summary)


if __name__ == "__main__":
    main()
