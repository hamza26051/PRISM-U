"""Deterministic CIFAR-10 split generation and validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from utils.io import load_yaml


def build_splits(seed, target_train_size, auxiliary_size, validation_size, forget_fraction):
    """Create deterministic disjoint partitions of 50,000 CIFAR-10 training indices."""
    total = target_train_size + auxiliary_size + validation_size
    if total != 50000:
        raise ValueError(f"CIFAR-10 training split has 50,000 samples, but requested {total}.")
    if not 0 < forget_fraction < 1:
        raise ValueError("forget_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    all_indices = np.arange(50000)
    rng.shuffle(all_indices)

    target = all_indices[:target_train_size]
    auxiliary = all_indices[target_train_size:target_train_size + auxiliary_size]
    validation = all_indices[target_train_size + auxiliary_size:]

    forget_n = int(round(target_train_size * forget_fraction))
    target_shuffled = target.copy()
    rng.shuffle(target_shuffled)
    forget = target_shuffled[:forget_n]
    retain = target_shuffled[forget_n:]

    return {
        "seed": int(seed),
        "target_train": target.tolist(),
        "forget": forget.tolist(),
        "retain": retain.tolist(),
        "auxiliary": auxiliary.tolist(),
        "validation": validation.tolist(),
        "official_test": list(range(10000)),
        "sizes": {
            "target_train": len(target),
            "forget": len(forget),
            "retain": len(retain),
            "auxiliary": len(auxiliary),
            "validation": len(validation),
            "official_test": 10000,
        },
    }


def load_split(path: Path) -> dict:
    """Load a split JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_split(split: dict, expected_sizes: Optional[dict] = None) -> dict:
    """Validate split disjointness, completeness, and sizes."""
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
        raise ValueError(f"Invalid split; failed checks: {bad}")
    return {"checks": checks, "sizes": {k: len(v) for k, v in arrays.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_stage1.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dcfg = cfg["data"]
    out = Path(dcfg["split_file"])
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not args.force:
        print(f"Split already exists: {out}")
        with open(out, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(existing["sizes"])
        return

    splits = build_splits(
        cfg["seed"],
        dcfg["target_train_size"],
        dcfg["auxiliary_size"],
        dcfg["validation_size"],
        dcfg["forget_fraction"],
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved split: {out}")
    print(splits["sizes"])


if __name__ == "__main__":
    main()
