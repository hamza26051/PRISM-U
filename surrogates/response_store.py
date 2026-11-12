"""Black-box response extraction and caching."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def blackbox_probabilities(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Extract softmax probability vectors from a model for all examples in a loader.

    This is the primary black-box interface: only the probability output is returned.
    No gradients, weights, or internal activations are exposed.

    Returns:
        np.ndarray of shape [N, num_classes] with float32 probabilities.
    """
    model.eval()
    chunks = []
    for images, _labels in loader:
        images = images.to(device, non_blocking=True)
        chunks.append(
            torch.softmax(model(images), dim=1)
            .cpu().numpy().astype(np.float32, copy=False)
        )
    return np.concatenate(chunks, axis=0)
