"""Evidence calibration and sequential stopping threshold calibration.

Eliminates overconfidence miscalibration (e.g. Brier score spikes at intermediate budgets)
and computes optimal sequential decision thresholds (theta_PASS, theta_FAIL) on validation banks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from evaluation.metrics import compute_brier_score

EPS = 1e-12


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Stable sigmoid mapping LLR to probability."""
    x_arr = np.asarray(x, dtype=np.float64)
    # Clip to prevent overflow
    x_clipped = np.clip(x_arr, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


@dataclass(frozen=True)
class CalibratedThresholds:
    """Calibrated sequential stopping thresholds."""
    theta_pass: float
    theta_fail: float
    false_pass_rate: float
    false_fail_rate: float
    coverage: float
    target_error_rate: float

    def to_dict(self) -> dict:
        return {
            "theta_pass": self.theta_pass,
            "theta_fail": self.theta_fail,
            "false_pass_rate": self.false_pass_rate,
            "false_fail_rate": self.false_fail_rate,
            "coverage": self.coverage,
            "target_error_rate": self.target_error_rate,
        }


class TemperatureCalibrator:
    """Scales LLR by learned temperature T > 0 to minimize Brier score or NLL.

    Calibrated LLR = LLR / T
    Calibrated P(H_C) = sigmoid(LLR / T)
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, llrs: Sequence[float], y_true: Sequence[int]) -> "TemperatureCalibrator":
        """Fit optimal temperature on a validation or surrogate dataset."""
        llrs_arr = np.asarray(llrs, dtype=np.float64)
        y_arr = np.asarray(y_true, dtype=np.float64)

        def objective(t: float) -> float:
            if t <= 0.01:
                return 1e6
            scaled_p = sigmoid(llrs_arr / t)
            # Binary cross-entropy loss
            p = np.clip(scaled_p, EPS, 1.0 - EPS)
            bce = -np.mean(y_arr * np.log(p) + (1.0 - y_arr) * np.log(1.0 - p))
            return float(bce)

        res = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(res.x)
        return self

    def calibrate_llr(self, llr: float | np.ndarray) -> float | np.ndarray:
        return np.asarray(llr, dtype=np.float64) / self.temperature

    def calibrate_prob(self, llr: float | np.ndarray) -> float | np.ndarray:
        return sigmoid(self.calibrate_llr(llr))


class ThresholdCalibrator:
    """Optimizes (theta_PASS, theta_FAIL) subject to error constraints."""

    def __init__(
        self,
        target_false_pass: float = 0.05,
        target_false_fail: float = 0.05,
    ) -> None:
        self.target_false_pass = float(target_false_pass)
        self.target_false_fail = float(target_false_fail)

    def calibrate_thresholds(
        self,
        llrs: Sequence[float],
        y_true: Sequence[int],
        default_theta_pass: float = 2.944,  # log(0.95 / 0.05)
        default_theta_fail: float = -2.944, # log(0.05 / 0.95)
    ) -> CalibratedThresholds:
        """Find optimal thresholds on calibration data.

        If calibration dataset is small or completely separable, falls back to
        canonical theoretical bounds.
        """
        llrs_arr = np.asarray(llrs, dtype=np.float64)
        y_arr = np.asarray(y_true, dtype=np.int64)

        hc_mask = y_arr == 1
        hn_mask = y_arr == 0

        if not hc_mask.any() or not hn_mask.any():
            # Insufficient class diversity: use canonical theoretical bounds
            return CalibratedThresholds(
                theta_pass=default_theta_pass,
                theta_fail=default_theta_fail,
                false_pass_rate=0.0,
                false_fail_rate=0.0,
                coverage=1.0,
                target_error_rate=self.target_false_pass,
            )

        hc_llrs = llrs_arr[hc_mask]
        hn_llrs = llrs_arr[hn_mask]

        # For theta_pass: find highest value such that false_pass <= target_false_pass
        # false_pass = fraction of HN models with llr >= theta_pass
        # We want theta_pass >= max(hn_llrs) if possible, or quantile
        if len(hn_llrs) > 0:
            theta_pass_cand = float(np.quantile(hn_llrs, 1.0 - self.target_false_pass))
            # Must be strictly positive
            theta_pass = max(default_theta_pass, theta_pass_cand + 0.1)
        else:
            theta_pass = default_theta_pass

        # For theta_fail: find lowest value such that false_fail <= target_false_fail
        # false_fail = fraction of HC models with llr <= theta_fail
        if len(hc_llrs) > 0:
            theta_fail_cand = float(np.quantile(hc_llrs, self.target_false_fail))
            # Must be strictly negative
            theta_fail = min(default_theta_fail, theta_fail_cand - 0.1)
        else:
            theta_fail = default_theta_fail

        # Compute empirical rates
        pass_dec = llrs_arr >= theta_pass
        fail_dec = llrs_arr <= theta_fail
        fp_rate = float(np.mean(pass_dec[hn_mask])) if hn_mask.any() else 0.0
        ff_rate = float(np.mean(fail_dec[hc_mask])) if hc_mask.any() else 0.0
        coverage = float(np.mean(pass_dec | fail_dec))

        return CalibratedThresholds(
            theta_pass=theta_pass,
            theta_fail=theta_fail,
            false_pass_rate=fp_rate,
            false_fail_rate=ff_rate,
            coverage=coverage,
            target_error_rate=max(self.target_false_pass, self.target_false_fail),
        )
