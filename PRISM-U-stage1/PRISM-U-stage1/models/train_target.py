import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from data.datasets import get_train_subset, get_test_dataset, make_loader
from models.resnet import build_resnet18_cifar
from utils.io import load_yaml, save_json
from utils.reproducibility import seed_everything


def accuracy_from_logits(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += y.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += y.size(0)
    return total_loss / total_n, total_correct / total_n


def main():
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

    best_path = ckpt_dir / f"{args.mode}_seed{cfg['seed']}.pt"

    print(f"Training mode={args.mode} split={split_name} samples={len(train_ds)} device={device}")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "mode": args.mode,
                "seed": cfg["seed"],
                "num_classes": cfg["model"]["num_classes"],
                "config": cfg,
                "best_val_acc": best_val,
                "epoch": epoch,
            }, best_path)

    elapsed = time.time() - start
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    summary = {
        "mode": args.mode,
        "seed": cfg["seed"],
        "train_split": split_name,
        "train_samples": len(train_ds),
        "epochs": epochs,
        "best_val_acc": best_val,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "training_seconds": elapsed,
        "checkpoint": str(best_path),
        "history": history,
    }
    save_json(summary, result_dir / f"train_{args.mode}_seed{cfg['seed']}.json")
    print(f"Saved checkpoint: {best_path}")
    print(f"Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
