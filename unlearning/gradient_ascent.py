"""Gradient Ascent unlearning: maximize loss on the forget set to remove memorization."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.optim import SGD

from data.datasets import make_loader
from models.resnet import load_model
from utils.reproducibility import seed_everything


def generate_gradient_ascent_family(
    *,
    seed: int,
    original_checkpoint: Path,
    forget_eval_dataset,
    output_dir: Path,
    device: str,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    total_steps: int,
    learning_rate: float,
    fractions: Sequence[int],
    force: bool = False,
) -> None:
    """Generate partial gradient-ascent unlearning checkpoints.

    Starting from original_checkpoint, applies gradient ascent (negated CE loss)
    on the forget set and saves checkpoints at specified fraction milestones
    (e.g., 25%, 50%, 75%, 100% of total_steps).

    Args:
        seed: Random seed for reproducibility.
        original_checkpoint: Path to the trained original model.
        forget_eval_dataset: PyTorch Dataset of forget-set examples (eval transform).
        output_dir: Directory to save ga{fraction}.pt checkpoints.
        device: 'cuda' or 'cpu'.
        num_classes: Number of output classes.
        batch_size: Batch size for gradient ascent.
        num_workers: DataLoader workers.
        total_steps: Total gradient ascent steps (100% = all steps).
        learning_rate: Learning rate for gradient ascent SGD.
        fractions: List of percentages at which to save checkpoints (e.g. [25, 50, 75, 100]).
        force: If True, regenerate even if checkpoints exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fractions = sorted({int(x) for x in fractions})

    if any(f <= 0 or f > 100 for f in fractions):
        raise ValueError("GA fractions must be in 1..100; original represents 0%.")

    paths = {f: output_dir / f"ga{f}.pt" for f in fractions}
    if all(p.exists() for p in paths.values()) and not force:
        print(f"[skip] GA family seed={seed}")
        return

    seed_everything(seed)
    model = load_model(original_checkpoint, device, num_classes)
    model.eval()  # Start from eval mode, then train during GA

    loader = make_loader(
        forget_eval_dataset, batch_size, True, num_workers,
        seed=seed, pin_memory=device.startswith("cuda"),
    )
    iterator = iter(loader)
    optimizer = SGD(model.parameters(), lr=float(learning_rate))
    criterion = nn.CrossEntropyLoss()

    # Map step numbers to fraction milestones
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
        # Negate the loss: gradient ASCENT to unlearn
        loss = -criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        if step in milestones:
            fraction = milestones[step]
            torch.save({
                "model_state": model.state_dict(),
                "mode": "gradient_ascent_unlearning",
                "seed": int(seed),
                "fraction": int(fraction),
                "steps_completed": int(step),
                "total_steps": int(total_steps),
                "learning_rate": float(learning_rate),
                "num_classes": int(num_classes),
                "source_checkpoint": str(original_checkpoint),
            }, paths[fraction])
            print(f"[GA seed={seed}] saved {fraction}% at step {step}")
