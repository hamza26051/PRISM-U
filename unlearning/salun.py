"""Saliency-based Unlearning (SalUn) implementation for PRISM-U target hardening.

Extends standard gradient ascent unlearning by restricting gradient updates
strictly to the top salient parameter fraction gamma on the forget set, preserving
retained representations and avoiding catastrophic collapse.

Reference:
    Fan et al., "SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency" (ICLR 2024).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.optim import SGD

from data.datasets import make_loader
from models.resnet import load_model
from utils.reproducibility import seed_everything


def compute_saliency_mask(
    model: nn.Module,
    forget_loader,
    device: str,
    saliency_fraction: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Compute binary parameter masks selecting top salient weights on the forget set.

    Args:
        model: Trained target model.
        forget_loader: DataLoader over the forget partition.
        device: 'cuda' or 'cpu'.
        saliency_fraction: Fraction gamma of top salient parameters to update (e.g. 0.01 for 1%).

    Returns:
        Dictionary mapping parameter names to binary {0, 1} boolean masks.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Accumulate absolute gradients
    grad_accum: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    n_batches = 0
    for images, labels in forget_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and name in grad_accum:
                    grad_accum[name].add_(param.grad.abs())

        n_batches += 1
        if n_batches >= 20:  # Bound saliency computation for speed
            break

    # Flatten all accumulated gradients to find global saliency threshold
    all_grads = torch.cat([g.view(-1) for g in grad_accum.values()])
    k = max(1, int(round(all_grads.numel() * saliency_fraction)))
    threshold = torch.topk(all_grads, k).values[-1]

    # Create binary masks
    masks = {
        name: (g >= threshold).to(device)
        for name, g in grad_accum.items()
    }
    return masks


def generate_salun_family(
    *,
    seed: int,
    original_checkpoint: Path,
    forget_eval_dataset,
    output_dir: Path,
    device: str,
    num_classes: int,
    batch_size: int = 128,
    num_workers: int = 0,
    total_steps: int = 100,
    learning_rate: float = 0.001,
    saliency_fraction: float = 0.01,
    fractions: Sequence[int] = (25, 50, 75, 100),
    force: bool = False,
) -> None:
    """Generate partial SalUn unlearning checkpoints (salun25, salun50, salun75, salun100)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fractions = sorted({int(x) for x in fractions})

    paths = {f: output_dir / f"salun{f}.pt" for f in fractions}
    if all(p.exists() for p in paths.values()) and not force:
        print(f"[skip] SalUn family seed={seed}")
        return

    seed_everything(seed)
    model = load_model(original_checkpoint, device, num_classes)

    loader = make_loader(
        forget_eval_dataset, batch_size, True, num_workers,
        seed=seed, pin_memory=device.startswith("cuda"),
    )

    print(f"[SalUn seed={seed}] Computing gradient saliency mask (gamma={saliency_fraction:.3f})...")
    masks = compute_saliency_mask(model, loader, device, saliency_fraction=saliency_fraction)

    model.train()
    iterator = iter(loader)
    optimizer = SGD(model.parameters(), lr=float(learning_rate))
    criterion = nn.CrossEntropyLoss()

    milestones = {max(1, int(total_steps) * f // 100): f for f in fractions}

    for step in range(1, int(total_steps) + 1):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Negative CE loss for gradient ascent
        loss = -criterion(model(images), labels)
        loss.backward()

        # Apply saliency mask to gradients before optimizer step
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and name in masks:
                    param.grad.mul_(masks[name])

        optimizer.step()

        if step in milestones:
            fraction = milestones[step]
            torch.save({
                "model_state": model.state_dict(),
                "mode": "salun_unlearning",
                "seed": int(seed),
                "fraction": int(fraction),
                "steps_completed": int(step),
                "total_steps": int(total_steps),
                "learning_rate": float(learning_rate),
                "saliency_fraction": float(saliency_fraction),
                "num_classes": int(num_classes),
                "source_checkpoint": str(original_checkpoint),
            }, paths[fraction])
            print(f"[SalUn seed={seed}] saved {fraction}% at step {step}")
