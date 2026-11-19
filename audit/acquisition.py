"""Acquisition functions and scoring policies for privacy-budgeted active auditing."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

EPS = 1e-8


class AcquisitionScorer(ABC):
    """Abstract scoring function combining compliance gain G and privacy risk delta_L."""

    @abstractmethod
    def score(
        self,
        gains: np.ndarray,
        risks: np.ndarray,
        step: int = 0,
        remaining_privacy_budget: float = float("inf"),
    ) -> np.ndarray:
        """Compute acquisition scores for candidate queries.

        Args:
            gains: Information gain values G_t(x | tau_t), shape [N].
            risks: Marginal privacy risk delta_L_t(x | tau_t) or static risk R(x), shape [N].
            step: Current audit step index t.
            remaining_privacy_budget: Remaining privacy budget B_P - L_hat(tau_t).

        Returns:
            Array of acquisition scores of shape [N] (higher is better).
        """


class RatioAcquisitionScorer(AcquisitionScorer):
    """Ratio acquisition: score(x) = G_t(x) / (max(0, delta_L_t(x)) + eps)."""

    def __init__(self, eps: float = 1e-4) -> None:
        self.eps = float(eps)

    def score(
        self,
        gains: np.ndarray,
        risks: np.ndarray,
        step: int = 0,
        remaining_privacy_budget: float = float("inf"),
    ) -> np.ndarray:
        g = np.asarray(gains, dtype=np.float64)
        r = np.asarray(risks, dtype=np.float64)
        # Non-negative effective risk penalty
        effective_r = np.maximum(0.0, r)
        return g / (effective_r + self.eps)


class AdditiveAcquisitionScorer(AcquisitionScorer):
    """Additive acquisition: score(x) = G_t(x) - lambda_t * delta_L_t(x)."""

    def __init__(self, penalty_lambda: float = 1.0, adaptive_penalty: bool = False) -> None:
        self.penalty_lambda = float(penalty_lambda)
        self.adaptive_penalty = bool(adaptive_penalty)

    def score(
        self,
        gains: np.ndarray,
        risks: np.ndarray,
        step: int = 0,
        remaining_privacy_budget: float = float("inf"),
    ) -> np.ndarray:
        g = np.asarray(gains, dtype=np.float64)
        r = np.asarray(risks, dtype=np.float64)

        if self.adaptive_penalty and np.isfinite(remaining_privacy_budget) and remaining_privacy_budget > 0:
            # Increase privacy weight as remaining privacy budget shrinks
            effective_lambda = self.penalty_lambda / max(remaining_privacy_budget, 1e-4)
        else:
            effective_lambda = self.penalty_lambda

        return g - effective_lambda * r
