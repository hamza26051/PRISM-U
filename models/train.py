"""Model training: from-scratch training with epoch-level resume support."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset
from tqdm import tqdm

from data.datasets import get_train_subset, get_test_dataset, make_loader
from models.resnet import build_resnet18_cifar, load_model
from evaluation.evaluate import evaluate
from utils.io import load_yaml, save_json
from utils.reproducibility import seed_everything


def train_from_scratch(
    *,
    seed: int,
    mode: str,
    indices: Sequence[int],
    train_aug_dataset,
    validation_loader,
    retain_loader,
    forget_loader,
    test_loader,
    output_path: Path,
    log_path: Path,
    device: str,
    num_classes: int,
    training_cfg: dict,
    num_workers: int,
    force: bool = False,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
) -> None:
    """Train a model from scratch with epoch-level checkpoint resume.

    Saves the best-validation-accuracy checkpoint to output_path.
    Maintains a .resume.pt file for crash recovery.
    """
    output_path, log_path = Path(output_path), Path(log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and log_path.exists() and not force:
        print(f"[skip] {mode} seed={seed}: {output_path}")
        return

    if mode not in {"original", "retrain"}:
        raise ValueError("mode must be original or retrain")

    seed_everything(seed)
    model = build_resnet18_cifar(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0))
    )
    optimizer = SGD(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        momentum=float(training_cfg["momentum"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=list(training_cfg["scheduler_milestones"]),
        gamma=float(training_cfg["scheduler_gamma"]),
    )

    epochs = int(training_cfg["epochs"])
    batch_size = int(training_cfg["batch_size"])
    resume_path = output_path.with_name(output_path.stem + ".resume.pt")
    start_epoch, best_val, history = 1, -1.0, []

    # Resume from crash checkpoint if available
    if resume_path.exists() and not force:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state["best_val"])
        history = list(state["history"])
        print(f"[resume] {mode} seed={seed} from epoch {start_epoch}")

    train_generator = torch.Generator()
    train_loader = make_loader(
        Subset(train_aug_dataset, list(indices)),
        batch_size,
        True,
        num_workers,
        seed=seed,
        pin_memory=device.startswith("cuda"),
        generator=train_generator,
    )

    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        train_generator.manual_seed(int(seed) + int(epoch))
        loader = train_loader
        model.train()
        total_loss = 0.0
        total_correct = total_n = 0

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += labels.size(0)

        val = evaluate(model, validation_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": float(total_loss / total_n),
            "train_acc": float(total_correct / total_n),
            "val_loss": val["cross_entropy"],
            "val_acc": val["accuracy"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        if val["accuracy"] > best_val:
            best_val = val["accuracy"]
            torch.save({
                "model_state": model.state_dict(),
                "mode": mode,
                "seed": int(seed),
                "num_classes": int(num_classes),
                "best_val_acc": float(best_val),
                "epoch": int(epoch),
            }, output_path)

        # Crash-recovery checkpoint
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "history": history,
        }, resume_path)

        print(
            f"[{mode} seed={seed}] epoch {epoch:02d}/{epochs} "
            f"train={row['train_acc']:.4f} val={row['val_acc']:.4f}",
            flush=True,
        )
        if on_epoch_end is not None:
            on_epoch_end({
                "model_label": f"{mode}_seed{seed}",
                "epoch": int(epoch),
                "epochs": int(epochs),
                "session_epochs": int(epoch - start_epoch + 1),
                "best_val": float(best_val),
                "elapsed_seconds": float(time.time() - started),
                "history": list(history),
            })

    # Final evaluation of best checkpoint
    best_model = load_model(output_path, device, num_classes)
    save_json({
        "mode": mode,
        "seed": int(seed),
        "train_samples": int(len(indices)),
        "epochs": epochs,
        "best_val_acc": float(best_val),
        "training_seconds": float(time.time() - started),
        "checkpoint": str(output_path),
        "sets": {
            "retain": evaluate(best_model, retain_loader, device),
            "forget": evaluate(best_model, forget_loader, device),
            "validation": evaluate(best_model, validation_loader, device),
            "test": evaluate(best_model, test_loader, device),
        },
        "history": history,
    }, log_path)

    if resume_path.exists():
        resume_path.unlink()


def main():
    """CLI entry point for standalone model training (Stage 1 style)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_stage1.yaml")
    parser.add_argument("--mode", choices=["original", "retrain"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tcfg, dcfg = cfg["training"], cfg["data"]
    split_name = "target_train" if args.mode == "original" else "retain"

    train_ds = get_train_subset(dcfg["root"], dcfg["split_file"], split_name, augment=True)
    val_ds = get_train_subset(dcfg["root"], dcfg["split_file"], "validation", augment=False)
    test_ds = get_test_dataset(dcfg["root"])

    train_loader = make_loader(train_ds, tcfg["batch_size"], True, dcfg["num_workers"])
    val_loader = make_loader(val_ds, cfg["evaluation"]["batch_size"], False, dcfg["num_workers"])
    test_loader = make_loader(test_ds, cfg["evaluation"]["batch_size"], False, dcfg["num_workers"])

    model = build_resnet18_cifar(cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))
    optimizer = SGD(
        model.parameters(),
        lr=tcfg["learning_rate"],
        momentum=tcfg["momentum"],
        weight_decay=tcfg["weight_decay"],
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=tcfg["scheduler_milestones"],
        gamma=tcfg["scheduler_gamma"],
    )

    epochs = args.epochs or tcfg["epochs"]
    best_val = -1.0
    start = time.time()
    history = []

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    result_dir = Path(cfg["paths"]["results"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{args.mode}_seed{cfg['seed']}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for x, y in tqdm(train_loader, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += y.size(0)

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        train_acc = total_correct / total_n
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_n,
            "train_acc": train_acc,
            "val_loss": val_metrics["cross_entropy"],
            "val_acc": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            torch.save({
                "model_state": model.state_dict(),
                "mode": args.mode,
                "seed": cfg["seed"],
                "num_classes": cfg["model"]["num_classes"],
                "best_val_acc": best_val,
                "epoch": epoch,
            }, ckpt_path)

        print(f"[{args.mode}] epoch {epoch:02d}/{epochs} "
              f"train={train_acc:.4f} val={val_metrics['accuracy']:.4f}")

    elapsed = time.time() - start
    save_json({
        "mode": args.mode,
        "seed": cfg["seed"],
        "epochs": epochs,
        "best_val_acc": best_val,
        "training_seconds": elapsed,
        "checkpoint": str(ckpt_path),
        "history": history,
    }, result_dir / f"train_{args.mode}_seed{cfg['seed']}.json")

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Training time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
