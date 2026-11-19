"""Unit tests for Phase 22 (SalUn saliency-based unlearning)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from unlearning.salun import compute_saliency_mask


def test_compute_saliency_mask():
    # Simple linear model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )

    # Synthetic dataset
    X = torch.randn(50, 10)
    y = torch.randint(0, 2, (50,))
    loader = DataLoader(TensorDataset(X, y), batch_size=10)

    gamma = 0.10  # 10% most salient
    masks = compute_saliency_mask(model, loader, device="cpu", saliency_fraction=gamma)

    assert "0.weight" in masks
    assert "2.weight" in masks

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_masked = sum(m.sum().item() for m in masks.values())

    # Selected parameters should be approximately 10%
    expected = int(round(total_params * gamma))
    assert abs(total_masked - expected) <= 2  # Allow for discrete rounding/ties
