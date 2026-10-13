import json
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _load_splits(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset(root, train, transform, download=True):
    return datasets.CIFAR10(root=root, train=train, transform=transform, download=download)


def get_train_subset(root, split_file, split_name, augment=True, download=True):
    split = _load_splits(split_file)
    tfm = train_transform() if augment else eval_transform()
    ds = get_dataset(root, train=True, transform=tfm, download=download)
    return Subset(ds, split[split_name])


def get_test_dataset(root, download=True):
    return get_dataset(root, train=False, transform=eval_transform(), download=download)


def make_loader(dataset, batch_size, shuffle, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
