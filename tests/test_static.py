from __future__ import annotations

import sys
from pathlib import Path

import yaml

STAGE2_DIR = Path(__file__).resolve().parents[1]
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from prismu_stage2.core import validate_split
from prismu_stage2.layout import build_model_manifest


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
    cfg = yaml.safe_load((STAGE2_DIR / "configs" / "stage2.yaml").read_text())
    s = set(cfg["bank"]["surrogate_seeds"])
    v = set(cfg["bank"]["validation_seeds"])
    f = {cfg["bank"]["final_seed"]}
    assert len(s) == 12
    assert len(v) == 4
    assert s.isdisjoint(v)
    assert s.isdisjoint(f)
    assert v.isdisjoint(f)


def test_default_manifest_shape_and_labels():
    cfg = yaml.safe_load((STAGE2_DIR / "configs" / "stage2.yaml").read_text())
    # Paths need not exist for manifest-shape testing.
    repo_root = STAGE2_DIR.parent
    manifest = build_model_manifest(cfg, repo_root)
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
