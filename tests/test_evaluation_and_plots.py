"""Unit tests for Pareto frontier extraction, plotting utilities, and result aggregation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from evaluation.compute_results import summarize_comparisons
from evaluation.pareto import ParetoPoint, compute_hypervolume, extract_pareto_frontier
from evaluation.plots import plot_feasibility_heatmap, plot_pareto_frontier, plot_risk_coverage
from utils.io import save_json


def test_pareto_frontier_extraction():
    pts = [
        ParetoPoint("p1", 10, 1.0, 0.90, 0.05, 0.60, 1.0, 10.0),
        ParetoPoint("p2", 10, 1.0, 0.85, 0.08, 0.55, 1.0, 10.0),
        ParetoPoint("p3", 10, 1.0, 0.70, 0.15, 0.70, 1.0, 10.0),  # Dominated by p1 (lower AUROC, higher leakage)
        ParetoPoint("p4", 10, 1.0, 0.95, 0.02, 0.75, 1.0, 10.0),
    ]
    frontier = extract_pareto_frontier(pts)
    # p3 is strictly dominated by p1 and p2
    frontier_names = [p.privacy_leakage for p in frontier]
    assert 0.70 not in frontier_names
    assert len(frontier) == 3
    # Check sorted ascending by leakage
    assert frontier[0].privacy_leakage == 0.55
    assert frontier[1].privacy_leakage == 0.60
    assert frontier[2].privacy_leakage == 0.75


def test_hypervolume():
    pts = [
        ParetoPoint("p1", 10, 1.0, 0.90, 0.05, 0.60, 1.0, 10.0),
        ParetoPoint("p2", 10, 1.0, 1.00, 0.01, 0.80, 1.0, 10.0),
    ]
    hv = compute_hypervolume(pts, ref_point=(0.5, 1.0))
    assert hv > 0.0


def test_plot_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Pareto plot
        pts = [
            ParetoPoint("random", 10, 1.0, 0.75, 0.15, 0.52, 1.0, 10.0),
            ParetoPoint("prism_u", 10, 0.60, 0.95, 0.02, 0.58, 1.0, 10.0),
        ]
        pareto_out = tmp_path / "pareto.png"
        plot_pareto_frontier(pts, pareto_out)
        assert pareto_out.exists() and pareto_out.stat().st_size > 1000

        # 2. Risk-coverage plot
        rc_pts = [
            {"coverage": 0.5, "error_rate": 0.01},
            {"coverage": 0.8, "error_rate": 0.03},
            {"coverage": 1.0, "error_rate": 0.05},
        ]
        rc_out = tmp_path / "risk_cov.png"
        plot_risk_coverage(rc_pts, rc_out)
        assert rc_out.exists() and rc_out.stat().st_size > 1000

        # 3. Feasibility heatmap
        mat = np.array([[0.5, 0.7], [0.8, 1.0]])
        heat_out = tmp_path / "heat.png"
        plot_feasibility_heatmap(mat, [5, 10], [0.6, 1.0], heat_out)
        assert heat_out.exists() and heat_out.stat().st_size > 1000


def test_summarize_comparisons():
    dummy = {
        "fixed_budget_comparisons": {
            "random": {
                "budgets": {
                    "5": {"auroc": 0.71, "auroc_ci_low": 0.65, "auroc_ci_high": 0.78, "brier_score": 0.15, "mean_leakage": 0.51}
                },
                "unique_trajectories": {"5": 5},
            }
        },
        "sequential_stopping_comparisons": {
            "full_prism_u": {
                "decision_metrics": {
                    "coverage": 1.0, "inconclusive_rate": 0.0,
                    "false_pass_rate": 0.0, "false_fail_rate": 0.0,
                    "conditional_accuracy": 1.0,
                },
                "auroc": 1.0,
                "mean_queries": 4.5,
                "median_queries": 4.0,
                "mean_leakage": 0.53,
            }
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "dummy.json"
        save_json(dummy, json_path)
        summary = summarize_comparisons(json_path)
        assert "Comparative Auditing Summary" in summary
        assert "random" in summary
        assert "full_prism_u" in summary


def test_summarize_ablations():
    dummy = {
        "A1_no_privacy": {
            "ablation_name": "A1: No Privacy Penalty",
            "decision_metrics": {
                "coverage": 1.0, "inconclusive_rate": 0.0,
                "false_pass_rate": 0.0, "false_fail_rate": 0.0,
                "conditional_accuracy": 1.0,
            },
            "auroc": 1.0,
            "mean_queries": 5.0,
            "mean_leakage": 0.90,
            "max_leakage": 0.95,
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "dummy_abl.json"
        save_json(dummy, json_path)
        from evaluation.compute_results import summarize_ablations
        summary = summarize_ablations(json_path)
        assert "Component Ablation Suite" in summary
        assert "A1: No Privacy Penalty" in summary
