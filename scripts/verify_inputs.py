from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

STAGE2_DIR = Path(__file__).resolve().parents[1]
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from prismu_stage2.core import assert_real_checkpoint, load_model, load_split, load_yaml, validate_split
from prismu_stage2.layout import resolved_paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="stage2/configs/stage2.yaml")
    args = p.parse_args()

    repo_root = Path.cwd().resolve()
    cfg = load_yaml(Path(args.config))
    paths = resolved_paths(cfg, repo_root)
    split = load_split(paths["stage1_split"])
    report = validate_split(split, {
        "target_train": 40000,
        "auxiliary": int(cfg["query_pool"]["expected_size"]),
        "validation": 5000,
    })
    assert_real_checkpoint(paths["stage1_original"])
    assert_real_checkpoint(paths["stage1_retrain"])

    # CPU-load both checkpoints to verify architecture/schema compatibility before
    # starting dozens of GPU training jobs.
    original = load_model(paths["stage1_original"], "cpu", int(cfg["model"]["num_classes"]))
    retrain = load_model(paths["stage1_retrain"], "cpu", int(cfg["model"]["num_classes"]))
    del original, retrain

    print("Stage 1 -> Stage 2 preflight: PASS")
    print("Split sizes:", report["sizes"])
    print("Original checkpoint:", paths["stage1_original"])
    print("Retrain checkpoint:", paths["stage1_retrain"])
    print("Artifact root:", paths["artifact_root"])


if __name__ == "__main__":
    main()
