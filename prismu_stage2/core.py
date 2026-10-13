from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (Path(repo_root) / p).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def checkpoint_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and checkpoint and all(isinstance(k, str) for k in checkpoint):
        return checkpoint
    raise ValueError("Unsupported checkpoint schema; expected model_state/state_dict.")


def assert_real_checkpoint(path: Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.stat().st_size < 1024:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "git-lfs.github.com/spec" in text:
            raise RuntimeError(f"{path} is a Git LFS pointer. Run `git lfs pull` first.")


def load_model(path: Path, device: str, num_classes: int = 10) -> nn.Module:
    assert_real_checkpoint(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_resnet18_cifar(num_classes)
    model.load_state_dict(checkpoint_state_dict(ckpt))
    return model.to(device)


def load_split(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_split(split: dict, expected_sizes: Optional[dict] = None) -> dict:
    required = ["target_train", "retain", "forget", "auxiliary", "validation"]
    missing = [k for k in required if k not in split]
    if missing:
        raise KeyError(f"Split file is missing keys: {missing}")
    arrays = {k: list(map(int, split[k])) for k in required}
    sets = {k: set(v) for k, v in arrays.items()}
    checks = {
        "retain_forget_disjoint": sets["retain"].isdisjoint(sets["forget"]),
        "retain_plus_forget_is_target": sets["retain"] | sets["forget"] == sets["target_train"],
        "auxiliary_disjoint_target": sets["auxiliary"].isdisjoint(sets["target_train"]),
        "validation_disjoint_target": sets["validation"].isdisjoint(sets["target_train"]),
        "auxiliary_disjoint_validation": sets["auxiliary"].isdisjoint(sets["validation"]),
        "no_duplicate_target": len(sets["target_train"]) == len(arrays["target_train"]),
        "no_duplicate_auxiliary": len(sets["auxiliary"]) == len(arrays["auxiliary"]),
        "no_duplicate_validation": len(sets["validation"]) == len(arrays["validation"]),
    }
    if expected_sizes:
        for key, expected in expected_sizes.items():
            if key in arrays and expected is not None:
                checks[f"size_{key}"] = len(arrays[key]) == int(expected)
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f"Invalid Stage 1 split; failed checks: {bad}")
    return {"checks": checks, "sizes": {k: len(v) for k, v in arrays.items()}}


def make_datasets(data_root: Path):
    data_root = str(Path(data_root))
    train_aug = datasets.CIFAR10(data_root, train=True, transform=train_transform(), download=True)
    train_eval = datasets.CIFAR10(data_root, train=True, transform=eval_transform(), download=True)
    test = datasets.CIFAR10(data_root, train=False, transform=eval_transform(), download=True)
    return train_aug, train_eval, test


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: Optional[int] = None, pin_memory: Optional[bool] = None) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
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


def train_from_scratch(*, seed: int, mode: str, indices: Sequence[int], train_aug_dataset, validation_loader: DataLoader, retain_loader: DataLoader, forget_loader: DataLoader, test_loader: DataLoader, output_path: Path, log_path: Path, device: str, num_classes: int, training_cfg: dict, num_workers: int, force: bool = False) -> None:
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
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training_cfg.get("label_smoothing", 0.0)))
    optimizer = SGD(model.parameters(), lr=float(training_cfg["learning_rate"]), momentum=float(training_cfg["momentum"]), weight_decay=float(training_cfg["weight_decay"]))
    scheduler = MultiStepLR(optimizer, milestones=list(training_cfg["scheduler_milestones"]), gamma=float(training_cfg["scheduler_gamma"]))
    epochs, batch_size = int(training_cfg["epochs"]), int(training_cfg["batch_size"])
    resume_path = output_path.with_name(output_path.stem + ".resume.pt")
    start_epoch, best_val, history = 1, -1.0, []
    if resume_path.exists() and not force:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state["best_val"])
        history = list(state["history"])
        print(f"[resume] {mode} seed={seed} from epoch {start_epoch}")
    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        loader = make_loader(Subset(train_aug_dataset, list(indices)), batch_size, True, num_workers, seed + epoch, device.startswith("cuda"))
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
        row = {"epoch": epoch, "train_loss": float(total_loss / total_n), "train_acc": float(total_correct / total_n), "val_loss": val["cross_entropy"], "val_acc": val["accuracy"], "lr": float(optimizer.param_groups[0]["lr"])}
        history.append(row)
        if val["accuracy"] > best_val:
            best_val = val["accuracy"]
            torch.save({"model_state": model.state_dict(), "mode": mode, "seed": int(seed), "num_classes": int(num_classes), "best_val_acc": float(best_val), "epoch": int(epoch), "stage": 2}, output_path)
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "epoch": int(epoch), "best_val": float(best_val), "history": history}, resume_path)
        print(f"[{mode} seed={seed}] epoch {epoch:02d}/{epochs} train={row['train_acc']:.4f} val={row['val_acc']:.4f}")
    best_model = load_model(output_path, device, num_classes)
    save_json({"mode": mode, "seed": int(seed), "train_samples": int(len(indices)), "epochs": epochs, "best_val_acc": float(best_val), "training_seconds_this_invocation": float(time.time() - started), "checkpoint": str(output_path), "sets": {"retain": evaluate(best_model, retain_loader, device), "forget": evaluate(best_model, forget_loader, device), "validation": evaluate(best_model, validation_loader, device), "test": evaluate(best_model, test_loader, device)}, "history": history}, log_path)
    if resume_path.exists():
        resume_path.unlink()


def generate_gradient_ascent_family(*, seed: int, original_checkpoint: Path, forget_eval_dataset, output_dir: Path, device: str, num_classes: int, batch_size: int, num_workers: int, total_steps: int, learning_rate: float, fractions: Sequence[int], force: bool = False) -> None:
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
    model.eval()
    loader = make_loader(forget_eval_dataset, batch_size, True, num_workers, seed, device.startswith("cuda"))
    iterator = iter(loader)
    optimizer, criterion = SGD(model.parameters(), lr=float(learning_rate)), nn.CrossEntropyLoss()
    milestones = {max(1, int(total_steps) * f // 100): f for f in fractions}
    for step in range(1, int(total_steps) + 1):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = -criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        if step in milestones:
            fraction = milestones[step]
            torch.save({"model_state": model.state_dict(), "mode": "gradient_ascent_unlearning", "seed": int(seed), "fraction": int(fraction), "steps_completed": int(step), "total_steps": int(total_steps), "learning_rate": float(learning_rate), "num_classes": int(num_classes), "source_checkpoint": str(original_checkpoint), "stage": 2}, paths[fraction])
            print(f"[GA seed={seed}] saved {fraction}% at step {step}")


@torch.no_grad()
def blackbox_probabilities(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    chunks = []
    for images, _labels in loader:
        images = images.to(device, non_blocking=True)
        chunks.append(torch.softmax(model(images), dim=1).cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)
