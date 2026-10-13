import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_stage1.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    py = sys.executable
    root = Path(__file__).resolve().parents[1]
    config = args.config

    run([py, "-m", "data.splits", "--config", config])

    base = ["--config", config]
    if args.epochs is not None:
        base += ["--epochs", str(args.epochs)]
    if args.device is not None:
        base += ["--device", args.device]

    run([py, "-m", "models.train_target", "--mode", "original", *base])
    run([py, "-m", "models.train_target", "--mode", "retrain", *base])

    # Read seed from the default naming convention. The evaluator also checks checkpoint metadata.
    import yaml
    with open(root / config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]

    for mode in ["original", "retrain"]:
        checkpoint = f"checkpoints/{mode}_seed{seed}.pt"
        run([py, "-m", "evaluation.evaluate", "--config", config, "--checkpoint", checkpoint, "--name", f"{mode}_seed{seed}"])

    print("\nStage 1 complete.")
    print("You now have the original target model, honest retrained reference, deterministic splits, and evaluation outputs.")


if __name__ == "__main__":
    main()
