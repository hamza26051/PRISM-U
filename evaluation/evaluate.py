"""Model evaluation: metrics collection and per-example probability dumping."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.datasets import get_train_subset, get_test_dataset, make_loader
from models.resnet import build_resnet18_cifar, load_model
from utils.io import load_yaml, save_json
from utils.reproducibility import seed_everything


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    """Evaluate a model on a DataLoader, returning accuracy, cross-entropy, confidence, entropy."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total = correct = 0
    total_loss = confidence_sum = entropy_sum = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        total_loss += criterion(logits, labels).item()
        correct += (probs.argmax(1) == labels).sum().item()
        confidence_sum += probs.max(dim=1).values.sum().item()
        entropy_sum += (-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)).sum().item()
        total += labels.size(0)

    return {
        "n": int(total),
        "accuracy": float(correct / total),
        "cross_entropy": float(total_loss / total),
        "mean_confidence": float(confidence_sum / total),
        "mean_entropy": float(entropy_sum / total),
    }


@torch.no_grad()
def collect_probabilities(model: nn.Module, loader: DataLoader, device: str):
    """Collect softmax probabilities and labels for all examples."""
    model.eval()
    probs_list, labels_list = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        labels_list.append(y.numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)


def main():
    """CLI entry point for checkpoint evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_stage1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dcfg = cfg["data"]

    model = load_model(args.checkpoint, device, cfg["model"]["num_classes"])

    eval_datasets = {
        "retain": get_train_subset(dcfg["root"], dcfg["split_file"], "retain", augment=False),
        "forget": get_train_subset(dcfg["root"], dcfg["split_file"], "forget", augment=False),
        "auxiliary": get_train_subset(dcfg["root"], dcfg["split_file"], "auxiliary", augment=False),
        "validation": get_train_subset(dcfg["root"], dcfg["split_file"], "validation", augment=False),
        "test": get_test_dataset(dcfg["root"]),
    }

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_mode": None,
        "checkpoint_seed": None,
        "sets": {},
    }
    prob_dump = {}

    for name, ds in eval_datasets.items():
        loader = make_loader(ds, cfg["evaluation"]["batch_size"], False, dcfg["num_workers"])
        probs, labels = collect_probabilities(model, loader, device)
        report["sets"][name] = evaluate(model, loader, device)
        print(name, report["sets"][name])
        if cfg["evaluation"].get("save_probabilities", True):
            prob_dump[f"{name}_probs"] = probs
            prob_dump[f"{name}_labels"] = labels

    out_name = args.name or Path(args.checkpoint).stem
    result_dir = Path(cfg["paths"]["results"])
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, result_dir / f"eval_{out_name}.json")
    if prob_dump:
        np.savez_compressed(result_dir / f"probs_{out_name}.npz", **prob_dump)
    print(f"Saved evaluation to {result_dir}")


if __name__ == "__main__":
    main()
