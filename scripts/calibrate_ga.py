"""Calibrate Gradient Ascent unlearning learning rate and step schedule."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Subset

from data.datasets import make_datasets, make_loader
from data.splits import load_split
from evaluation.evaluate import evaluate
from models.resnet import load_model
from utils.io import load_yaml, save_json
from utils.reproducibility import seed_everything


def run_sweep(
    original_checkpoint: Path,
    split_file: Path,
    data_root: Path,
    candidate_lrs: list[float],
    total_steps: int = 100,
    fractions: list[int] = [25, 50, 75, 100],
    batch_size: int = 128,
    num_classes: int = 10,
    device: str = "cuda",
    seed: int = 42,
):
    print("=" * 80)
    print("PRISM-U Gradient Ascent Unlearning Calibration Sweep")
    print(f"Device: {device} | Base: {original_checkpoint} | Candidates: {candidate_lrs}")
    print("=" * 80)

    split = load_split(split_file)
    _train_aug, train_eval, test_data = make_datasets(data_root)

    forget_eval_dataset = Subset(train_eval, split["forget"])
    retain_loader = make_loader(Subset(train_eval, split["retain"]), 256, False, 2, pin_memory=device.startswith("cuda"))
    forget_loader = make_loader(forget_eval_dataset, 256, False, 2, pin_memory=device.startswith("cuda"))
    test_loader = make_loader(test_data, 256, False, 2, pin_memory=device.startswith("cuda"))

    # Baseline original evaluation
    orig_model = load_model(original_checkpoint, device, num_classes)
    base_forget = evaluate(orig_model, forget_loader, device)
    base_retain = evaluate(orig_model, retain_loader, device)
    base_test = evaluate(orig_model, test_loader, device)
    del orig_model

    print(f"Base Original -> Forget Acc: {base_forget['accuracy']*100:.2f}%, Retain Acc: {base_retain['accuracy']*100:.2f}%, Test Acc: {base_test['accuracy']*100:.2f}%")

    results = {
        "baseline_original": {
            "forget": base_forget,
            "retain": base_retain,
            "test": base_test,
        },
        "candidates": {},
    }

    milestones = {max(1, int(total_steps) * f // 100): f for f in fractions}

    for lr in candidate_lrs:
        lr_key = f"lr_{lr}"
        print(f"\n--- Testing LR = {lr} ---")
        seed_everything(seed)
        model = load_model(original_checkpoint, device, num_classes)
        model.eval()

        loader = make_loader(forget_eval_dataset, batch_size, True, 2, seed=seed, pin_memory=device.startswith("cuda"))
        iterator = iter(loader)
        optimizer = SGD(model.parameters(), lr=float(lr))
        criterion = nn.CrossEntropyLoss()

        results["candidates"][lr_key] = {"lr": lr, "milestones": {}}

        start_time = time.time()
        for step in range(1, total_steps + 1):
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, labels = next(iterator)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = -criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            if step in milestones:
                fraction = milestones[step]
                f_metrics = evaluate(model, forget_loader, device)
                r_metrics = evaluate(model, retain_loader, device)
                t_metrics = evaluate(model, test_loader, device)

                results["candidates"][lr_key]["milestones"][f"step_{step}_{fraction}pct"] = {
                    "step": step,
                    "fraction": fraction,
                    "forget": f_metrics,
                    "retain": r_metrics,
                    "test": t_metrics,
                }
                print(f"  Step {step:3d} ({fraction:3d}%): "
                      f"Forget Acc={f_metrics['accuracy']*100:6.2f}% (Loss={f_metrics['cross_entropy']:.4f}) | "
                      f"Retain Acc={r_metrics['accuracy']*100:6.2f}% (Loss={r_metrics['cross_entropy']:.4f}) | "
                      f"Test Acc={t_metrics['accuracy']*100:6.2f}%")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    p = argparse.ArgumentParser(description="Calibrate GA learning rate on RTX 4000 Ada GPU.")
    p.add_argument("--config", default="configs/stage2.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--lrs", nargs="+", type=float, default=[0.0002, 0.0005, 0.001, 0.002, 0.005])
    args = p.parse_args()

    cfg = load_yaml(Path(args.config))
    repo_root = Path.cwd().resolve()
    paths = cfg["paths"]

    results = run_sweep(
        original_checkpoint=Path(paths["stage1_original"]),
        split_file=Path(paths["stage1_split"]),
        data_root=Path(paths["data_root"]),
        candidate_lrs=args.lrs,
        total_steps=int(cfg["unlearning"]["total_steps"]),
        fractions=cfg["unlearning"]["fractions"],
        batch_size=int(cfg["unlearning"]["batch_size"]),
        num_classes=int(cfg["model"]["num_classes"]),
        device=args.device,
        seed=int(cfg["bank"]["final_seed"]),
    )

    out_file = Path(paths["artifact_root"]) / "diagnostics" / "ga_calibration_sweep.json"
    save_json(results, out_file)
    print(f"\nCalibration sweep saved to: {out_file}")


if __name__ == "__main__":
    main()

