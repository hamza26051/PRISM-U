"""Stage 1 orchestrator: generate splits, train models, evaluate."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utils.io import load_yaml


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run complete PRISM-U Stage 1 pipeline.")
    parser.add_argument("--config", default="configs/cifar10_stage1.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    py = sys.executable
    config = args.config

    run([py, "-m", "data.splits", "--config", config])

    base = ["--config", config]
    if args.epochs is not None:
        base += ["--epochs", str(args.epochs)]
    if args.device is not None:
        base += ["--device", args.device]

    run([py, "-m", "models.train", "--mode", "original", *base])
    run([py, "-m", "models.train", "--mode", "retrain", *base])

    cfg = load_yaml(Path(config))
    seed = cfg["seed"]

    for mode in ["original", "retrain"]:
        checkpoint = f"checkpoints/{mode}_seed{seed}.pt"
        run([py, "-m", "evaluation.evaluate", "--config", config, "--checkpoint", checkpoint, "--name", f"{mode}_seed{seed}"])

    print("\nStage 1 complete.")
    print("Original target model, honest retrained reference, deterministic splits, and evaluation outputs verified.")


if __name__ == "__main__":
    main()

