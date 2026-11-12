"""Static tests for split validation, seed isolation, and manifest schema."""
from __future__ import annotations

from pathlib import Path

import yaml

from data.splits import validate_split
from surrogates.bank import build_model_manifest

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_split_validation_accepts_disjoint_partition():
    split = {
        "target_train": [0, 1, 2, 3],
        "retain": [0, 1, 2],
        "forget": [3],
        "auxiliary": [4, 5],
        "validation": [6, 7],
    }
    report = validate_split(split, {"target_train": 4, "auxiliary": 2, "validation": 2})
    assert all(report["checks"].values())


def test_default_config_has_isolated_seed_groups():
    cfg = yaml.safe_load((ROOT_DIR / "configs" / "stage2.yaml").read_text())
    s = set(cfg["bank"]["surrogate_seeds"])
    v = set(cfg["bank"]["validation_seeds"])
    f = {cfg["bank"]["final_seed"]}
    assert len(s) == 12
    assert len(v) == 4
    assert s.isdisjoint(v)
    assert s.isdisjoint(f)
    assert v.isdisjoint(f)


def test_default_manifest_shape_and_labels():
    cfg = yaml.safe_load((ROOT_DIR / "configs" / "stage2.yaml").read_text())
    manifest = build_model_manifest(cfg, ROOT_DIR)
    surrogate = manifest[manifest.role == "surrogate"]
    validation = manifest[manifest.role == "validation"]
    final = manifest[manifest.role == "final"]
    assert len(surrogate) == 72
    assert len(validation) == 24
    assert len(final) == 6
    assert (surrogate.hypothesis == "HC").sum() == 12
    assert (surrogate.hypothesis == "HN").sum() == 48
    assert (surrogate.hypothesis == "EXPERIMENTAL").sum() == 12
    assert (validation.hypothesis == "HC").sum() == 4
    assert (validation.hypothesis == "HN").sum() == 16

