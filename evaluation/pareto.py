"""Pareto frontier extraction and trade-off analysis between audit accuracy and privacy leakage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ParetoPoint:
    """Single operating point on an audit-privacy trade-off curve."""
    policy_name: str
    query_budget: int
    privacy_budget: float
    audit_auroc: float
    audit_brier: float
    privacy_leakage: float  # Retained-set MIA AUROC or L_hat
    coverage: float
    queries_used: float

    def to_dict(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "query_budget": self.query_budget,
            "privacy_budget": self.privacy_budget,
            "audit_auroc": self.audit_auroc,
            "audit_brier": self.audit_brier,
            "privacy_leakage": self.privacy_leakage,
            "coverage": self.coverage,
            "queries_used": self.queries_used,
        }


def extract_pareto_frontier(
    points: Sequence[ParetoPoint],
    maximize_audit: bool = True,
    minimize_leakage: bool = True,
) -> list[ParetoPoint]:
    """Extract non-dominated Pareto-optimal points.

    A point A dominates point B if:
    - audit_auroc(A) >= audit_auroc(B) (and strictly greater in at least one objective)
    - privacy_leakage(A) <= privacy_leakage(B)
    """
    pts = list(points)
    if not pts:
        return []

    # Sort primarily by leakage (ascending), secondarily by audit AUROC (descending)
    pts.sort(key=lambda p: (p.privacy_leakage, -p.audit_auroc))

    frontier: list[ParetoPoint] = []
    best_audit_so_far = -float("inf")

    for p in pts:
        if p.audit_auroc > best_audit_so_far:
            frontier.append(p)
            best_audit_so_far = p.audit_auroc

    return frontier


def compute_hypervolume(
    frontier: Sequence[ParetoPoint],
    ref_point: tuple[float, float] = (0.5, 1.0),
) -> float:
    """Compute 2D hypervolume dominated by the Pareto frontier.

    Objectives:
    - x = Privacy Leakage in [0.5, 1.0] (minimize)
    - y = Audit AUROC in [0.5, 1.0] (maximize)

    ref_point is (max_leakage_ref=1.0, min_audit_ref=0.5).
    """
    if not frontier:
        return 0.0

    ref_audit, ref_leakage = ref_point
    # Sort by leakage ascending
    pts = sorted(frontier, key=lambda p: p.privacy_leakage)

    # Compute step-wise rectangular area under the Pareto curve
    hv = 0.0
    for i in range(len(pts)):
        leak_i = pts[i].privacy_leakage
        next_leak = pts[i + 1].privacy_leakage if i + 1 < len(pts) else ref_leakage
        width = max(0.0, next_leak - leak_i)
        height = max(0.0, pts[i].audit_auroc - ref_audit)
        hv += width * height

    return float(hv)
