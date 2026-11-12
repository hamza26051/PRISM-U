"""CIFAR-adapted ResNet-18 builder and model loading utilities."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

from utils.io import assert_real_checkpoint, checkpoint_state_dict


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """Build a ResNet-18 adapted for 32x32 CIFAR images.

    Modifications from ImageNet ResNet-18:
    - conv1: 3x3 kernel, stride 1, padding 1 (instead of 7x7, stride 2)
    - maxpool replaced with Identity
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def load_model(path: Path, device: str, num_classes: int = 10) -> nn.Module:
    """Load a model checkpoint, verifying it is not a Git LFS pointer."""
    assert_real_checkpoint(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_resnet18_cifar(num_classes)
    model.load_state_dict(checkpoint_state_dict(ckpt))
    return model.to(device)
