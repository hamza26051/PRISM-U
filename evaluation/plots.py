"""Publication-grade visualization suite for PRISM-U evaluation results.

Generates:
1. Pareto Frontier: Audit AUROC vs. Empirical Retained-Set Privacy Leakage.
2. Risk-Coverage Curves: Error rate among decided cases vs. Coverage across thresholds.
3. (B_P, B_Q) Feasibility Phase Diagram: 2D heatmap of reliable verification vs. abstention.
4. LLR Trajectories: Step-by-step sequential evidence accumulation with decision boundaries.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence

# Windows font cache safety
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prism_u_mpl"))

import matplotlib.pyplot as plt
import numpy as np

from evaluation.pareto import ParetoPoint, extract_pareto_frontier


def plot_pareto_frontier(
    points: Sequence[ParetoPoint],
    output_path: Path,
    title: str = "Empirical Audit-Privacy Pareto Frontier",
) -> None:
    """Plot Audit AUROC vs. Empirical Privacy Leakage with Pareto frontier highlighted."""
    pts = list(points)
    if not pts:
        return

    frontier = extract_pareto_frontier(pts)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Group points by policy
    policies = sorted({p.policy_name for p in pts})
    markers = ["o", "s", "^", "D", "v", "p", "*"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(policies))))

    for i, pol in enumerate(policies):
        pol_pts = [p for p in pts if p.policy_name == pol]
        x = [p.privacy_leakage for p in pol_pts]
        y = [p.audit_auroc for p in pol_pts]
        ax.scatter(
            x, y,
            label=pol,
            marker=markers[i % len(markers)],
            color=colors[i],
            alpha=0.7,
            s=50,
            edgecolors="none",
        )

    # Plot Pareto frontier
    if frontier:
        f_x = [p.privacy_leakage for p in frontier]
        f_y = [p.audit_auroc for p in frontier]
        ax.plot(
            f_x, f_y,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Pareto Frontier",
            zorder=10,
        )
        ax.scatter(f_x, f_y, color="black", s=80, marker="*", zorder=11)

    ax.set_xlabel("Retained-Set Privacy Leakage (MIA AUROC / L_hat)", fontsize=12)
    ax.set_ylabel("Compliance Audit AUROC", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.48, 1.02)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_risk_coverage(
    coverage_points: list[dict],
    output_path: Path,
    title: str = "Risk-Coverage Trade-off Curve",
) -> None:
    """Plot Error Rate among decided targets vs. Coverage (fraction of decided cases).

    Args:
        coverage_points: List of dicts with 'coverage' and 'error_rate' (1 - conditional_acc).
    """
    if not coverage_points:
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    # Sort by coverage ascending
    pts = sorted(coverage_points, key=lambda p: p["coverage"])
    cov = [p["coverage"] for p in pts]
    err = [p["error_rate"] for p in pts]

    ax.plot(cov, err, marker="o", color="#d62728", linewidth=2, label="PRISM-U")
    ax.set_xlabel("Coverage (Fraction Decided)", fontsize=12)
    ax.set_ylabel("Selective Error Rate (on Decided Cases)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(-0.02, max(err) * 1.2 if err else 1.0)
    ax.legend(loc="upper left")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_feasibility_heatmap(
    grid_matrix: np.ndarray,
    bq_values: Sequence[int],
    bp_values: Sequence[float],
    output_path: Path,
    metric_label: str = "Audit AUROC",
    title: str = "(B_P, B_Q) Feasibility Phase Diagram",
) -> None:
    """Plot 2D feasibility heatmap across (B_P, B_Q) budget sweeps."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    im = ax.imshow(
        grid_matrix,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        vmin=0.5,
        vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label, fontsize=11)

    ax.set_xticks(range(len(bq_values)))
    ax.set_xticklabels([str(b) for b in bq_values], fontsize=10)
    ax.set_yticks(range(len(bp_values)))
    ax.set_yticklabels([f"{p:.2f}" if np.isfinite(p) else "inf" for p in bp_values], fontsize=10)

    ax.set_xlabel("Query Budget B_Q", fontsize=12)
    ax.set_ylabel("Privacy Budget B_P", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Annotate values in cells
    for i in range(len(bp_values)):
        for j in range(len(bq_values)):
            val = grid_matrix[i, j]
            color = "white" if val < 0.75 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
