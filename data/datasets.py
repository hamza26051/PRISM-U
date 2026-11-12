"""CIFAR-10 dataset helpers, transforms, and DataLoader factory."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from utils.reproducibility import seed_worker

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def train_transform():
    """Standard CIFAR-10 training augmentation."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def eval_transform():
    """Deterministic CIFAR-10 evaluation transform."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def make_datasets(data_root: Path):
    """Return (train_aug, train_eval, test) CIFAR-10 dataset objects."""
    data_root = str(Path(data_root))
    train_aug = datasets.CIFAR10(data_root, train=True, transform=train_transform(), download=True)
    train_eval = datasets.CIFAR10(data_root, train=True, transform=eval_transform(), download=True)
    test = datasets.CIFAR10(data_root, train=False, transform=eval_transform(), download=True)
    return train_aug, train_eval, test


def get_dataset(root, train, transform, download=True):
    """Load a CIFAR-10 dataset split."""
    return datasets.CIFAR10(root=root, train=train, transform=transform, download=download)


def get_train_subset(root, split_file, split_name, augment=True, download=True):
    """Load a subset of CIFAR-10 training data based on a split file."""
    from data.splits import load_split
    split = load_split(split_file)
    tfm = train_transform() if augment else eval_transform()
    ds = get_dataset(root, train=True, transform=tfm, download=download)
    return Subset(ds, split[split_name])


def get_test_dataset(root, download=True):
    """Load the official CIFAR-10 test set."""
    return get_dataset(root, train=False, transform=eval_transform(), download=download)


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    seed: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Create a DataLoader with optional reproducible seeding."""
    if generator is None and seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    elif generator is not None and seed is not None:
        generator.manual_seed(int(seed))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
        generator=generator,
        worker_init_fn=seed_worker if int(num_workers) > 0 else None,
    )
