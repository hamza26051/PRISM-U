"""Membership Inference Attack (MIA) methods for retained-set leakage evaluation.

Implements metric-based and learned attacks operating purely on black-box model
outputs (softmax probability vectors). No model internals (gradients, weights,
activations) are accessed.

Attack taxonomy:
    1. Metric-based (threshold) attacks:
       - LossAttack: Cross-entropy loss of predicted probabilities against ground-truth label.
       - EntropyAttack: Prediction entropy H(p(y|x)).
       - ConfidenceAttack: Max predicted class probability (top-1 confidence).
       - MarginAttack: Gap between top-2 predicted class probabilities.
    2. Learned (model-based) attacks:
       - LogisticRegressionAttack: LR classifier on CLR-transformed probability vectors.
       - MLPAttack: 2-hidden-layer MLP on CLR-transformed probability vectors.

Convention:
    - Members (retained training data D_r) → label 1.
    - Non-members (held-out data) → label 0.
    - Higher attack score → more likely to be a member.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from audit.evidence import to_clr

EPS = 1e-12


@dataclass(frozen=True)
class AttackOutput:
    """Output from a membership inference attack on a set of samples.

    Attributes:
        scores: Per-sample membership scores (higher → more likely member).
        predictions: Optional binary predictions if a threshold is applied.
        attack_name: Name identifier of the attack method.
    """
    scores: np.ndarray
    predictions: Optional[np.ndarray]
    attack_name: str

    def __post_init__(self) -> None:
        s = np.asarray(self.scores, dtype=np.float64).copy()
        s.setflags(write=False)
        object.__setattr__(self, "scores", s)
        if self.predictions is not None:
            p = np.asarray(self.predictions, dtype=np.int64).copy()
            p.setflags(write=False)
            object.__setattr__(self, "predictions", p)


class MembershipInferenceAttack(ABC):
    """Abstract base class for all membership inference attacks."""

    name: str = "base"

    @abstractmethod
    def attack(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> AttackOutput:
        """Compute membership scores for a batch of probability vectors.

        Args:
            probs: Softmax probability vectors of shape [N, K] where K = num_classes.
            labels: Optional ground-truth class labels of shape [N] (required for
                    loss-based attacks, ignored by entropy/confidence/margin attacks).

        Returns:
            AttackOutput with per-sample membership scores.
        """

    def fit(
        self,
        member_probs: np.ndarray,
        nonmember_probs: np.ndarray,
        member_labels: Optional[np.ndarray] = None,
        nonmember_labels: Optional[np.ndarray] = None,
    ) -> "MembershipInferenceAttack":
        """Train/calibrate attack if needed (no-op for metric-based attacks)."""
        return self


# ---------------------------------------------------------------------------
# Metric-based (threshold) attacks
# ---------------------------------------------------------------------------

class LossAttack(MembershipInferenceAttack):
    """Cross-entropy loss attack: members tend to have lower loss (higher negative loss score).

    Score = -loss(p(y|x), y_true), so higher score → lower loss → more likely member.
    """

    name = "loss"

    def attack(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> AttackOutput:
        if labels is None:
            raise ValueError("LossAttack requires ground-truth labels.")
        p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0)
        y = np.asarray(labels, dtype=np.int64)
        if p.ndim != 2:
            raise ValueError(f"Expected 2D probs [N, K], got shape {p.shape}")
        if y.shape[0] != p.shape[0]:
            raise ValueError(f"Labels length {y.shape[0]} != probs length {p.shape[0]}")

        # Cross-entropy loss per sample
        n = p.shape[0]
        ce_loss = -np.log(p[np.arange(n), y])
        # Negate: higher score → more likely member (lower loss)
        scores = -ce_loss
        return AttackOutput(scores=scores, predictions=None, attack_name=self.name)


class EntropyAttack(MembershipInferenceAttack):
    """Prediction entropy attack: members tend to have lower entropy (more confident).

    Score = -H(p(y|x)), so higher score → lower entropy → more likely member.
    """

    name = "entropy"

    def attack(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> AttackOutput:
        p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0)
        if p.ndim != 2:
            raise ValueError(f"Expected 2D probs [N, K], got shape {p.shape}")
        entropy = -np.sum(p * np.log(p), axis=1)
        scores = -entropy  # Negate: lower entropy → higher score → more likely member
        return AttackOutput(scores=scores, predictions=None, attack_name=self.name)


class ConfidenceAttack(MembershipInferenceAttack):
    """Top-1 confidence attack: members tend to have higher max predicted probability.

    Score = max_k p(y=k|x).
    """

    name = "confidence"

    def attack(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> AttackOutput:
        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 2:
            raise ValueError(f"Expected 2D probs [N, K], got shape {p.shape}")
        scores = np.max(p, axis=1)
        return AttackOutput(scores=scores, predictions=None, attack_name=self.name)


class MarginAttack(MembershipInferenceAttack):
    """Margin attack: members tend to have larger gap between top-2 predictions.

    Score = p(y=k_1|x) - p(y=k_2|x) where k_1 > k_2 are the top-2 predicted classes.
    """

    name = "margin"

    def attack(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> AttackOutput:
        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 2:
            raise ValueError(f"Expected 2D probs [N, K], got shape {p.shape}")
        sorted_p = np.sort(p, axis=1)[:, ::-1]
        scores = sorted_p[:, 0] - sorted_p[:, 1]
        return AttackOutput(scores=scores, predictions=None, attack_name=self.name)


# ---------------------------------------------------------------------------
# Learned (model-based) attacks operating on CLR representations
# ---------------------------------------------------------------------------

class LogisticRegressionAttack(MembershipInferenceAttack):
    """Logistic regression classifier on CLR-transformed probability vectors.

    Training partitions are distinct from evaluation partitions (enforced externally).
    """

    name = "logistic_regression"

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 42) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self._clf = None

    def _to_features(self, probs: np.ndarray) -> np.ndarray:
        """Transform probabilities to CLR features, appending entropy and max-confidence."""
        clr = to_clr(probs)
        p = np.clip(probs, EPS, 1.0)
        entropy = -np.sum(p * np.log(p), axis=1, keepdims=True)
        max_conf = np.max(probs, axis=1, keepdims=True)
        return np.hstack([clr, entropy, max_conf])

    def fit(
        self,
        member_probs: np.ndarray,
        nonmember_probs: np.ndarray,
        member_labels: Optional[np.ndarray] = None,
        nonmember_labels: Optional[np.ndarray] = None,
    ) -> "LogisticRegressionAttack":
        from sklearn.linear_model import LogisticRegression

        X_mem = self._to_features(member_probs)
        X_nonmem = self._to_features(nonmember_probs)
        X = np.vstack([X_mem, X_nonmem])
        y = np.concatenate([np.ones(len(X_mem)), np.zeros(len(X_nonmem))])

        self._clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.seed,
            solver="lbfgs",
        )
        self._clf.fit(X, y)
        return self

    def attack(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> AttackOutput:
        if self._clf is None:
            raise RuntimeError("LogisticRegressionAttack must be fit() before attack().")
        X = self._to_features(probs)
        scores = self._clf.predict_proba(X)[:, 1]
        predictions = self._clf.predict(X).astype(np.int64)
        return AttackOutput(scores=scores, predictions=predictions, attack_name=self.name)


class MLPAttack(MembershipInferenceAttack):
    """Two-hidden-layer MLP classifier on CLR-transformed probability vectors.

    Architecture: Input → 64 → 32 → 1 (sigmoid) with ReLU activations.
    Uses sklearn MLPClassifier for portability (no torch dependency in audit code).
    """

    name = "mlp"

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (64, 32),
        max_iter: int = 500,
        alpha: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self._clf = None

    def _to_features(self, probs: np.ndarray) -> np.ndarray:
        clr = to_clr(probs)
        p = np.clip(probs, EPS, 1.0)
        entropy = -np.sum(p * np.log(p), axis=1, keepdims=True)
        max_conf = np.max(probs, axis=1, keepdims=True)
        margin = np.sort(probs, axis=1)[:, -1:] - np.sort(probs, axis=1)[:, -2:-1]
        return np.hstack([clr, entropy, max_conf, margin])

    def fit(
        self,
        member_probs: np.ndarray,
        nonmember_probs: np.ndarray,
        member_labels: Optional[np.ndarray] = None,
        nonmember_labels: Optional[np.ndarray] = None,
    ) -> "MLPAttack":
        from sklearn.neural_network import MLPClassifier

        X_mem = self._to_features(member_probs)
        X_nonmem = self._to_features(nonmember_probs)
        X = np.vstack([X_mem, X_nonmem])
        y = np.concatenate([np.ones(len(X_mem)), np.zeros(len(X_nonmem))])

        self._clf = MLPClassifier(
            hidden_layer_sizes=self.hidden_sizes,
            max_iter=self.max_iter,
            alpha=self.alpha,
            random_state=self.seed,
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.15,
        )
        self._clf.fit(X, y)
        return self

    def attack(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> AttackOutput:
        if self._clf is None:
            raise RuntimeError("MLPAttack must be fit() before attack().")
        X = self._to_features(probs)
        scores = self._clf.predict_proba(X)[:, 1]
        predictions = self._clf.predict(X).astype(np.int64)
        return AttackOutput(scores=scores, predictions=predictions, attack_name=self.name)


# ---------------------------------------------------------------------------
# Registry of all available attack methods
# ---------------------------------------------------------------------------

ALL_METRIC_ATTACKS = {
    "loss": LossAttack,
    "entropy": EntropyAttack,
    "confidence": ConfidenceAttack,
    "margin": MarginAttack,
}

ALL_LEARNED_ATTACKS = {
    "logistic_regression": LogisticRegressionAttack,
    "mlp": MLPAttack,
}

ALL_ATTACKS = {**ALL_METRIC_ATTACKS, **ALL_LEARNED_ATTACKS}
