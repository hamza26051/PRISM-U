from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .core import resolve_repo_path


def resolved_paths(cfg: dict, repo_root: Path) -> dict:
    return {k: resolve_repo_path(repo_root, v) for k, v in cfg["paths"].items()}


def family_dir(artifact_root: Path, role: str, seed: int) -> Path:
    return Path(artifact_root) / "checkpoints" / role / f"seed{seed}"


def original_path(artifact_root: Path, role: str, seed: int, stage1_original: Path, final_seed: int) -> Path:
    if role == "final" and int(seed) == int(final_seed):
        return Path(stage1_original)
    return family_dir(artifact_root, role, seed) / "original.pt"


def retrain_path(artifact_root: Path, role: str, seed: int, stage1_retrain: Path, final_seed: int) -> Path:
    if role == "final" and int(seed) == int(final_seed):
        return Path(stage1_retrain)
    return family_dir(artifact_root, role, seed) / "retrain.pt"


def ga_path(artifact_root: Path, role: str, seed: int, fraction: int) -> Path:
    return family_dir(artifact_root, role, seed) / "ga" / f"ga{int(fraction)}.pt"


def iter_families(cfg: dict):
    for seed in cfg["bank"]["surrogate_seeds"]:
        yield "surrogate", int(seed)
    for seed in cfg["bank"]["validation_seeds"]:
        yield "validation", int(seed)
    yield "final", int(cfg["bank"]["final_seed"])


def build_model_manifest(cfg: dict, repo_root: Path) -> pd.DataFrame:
    paths = resolved_paths(cfg, repo_root)
    artifact_root = paths["artifact_root"]
    final_seed = int(cfg["bank"]["final_seed"])
    rows = []
    for role, seed in iter_families(cfg):
        entries = [
            ("retrain", 100, "HC", retrain_path(artifact_root, role, seed, paths["stage1_retrain"], final_seed)),
            ("original", 0, "HN", original_path(artifact_root, role, seed, paths["stage1_original"], final_seed)),
            ("ga25", 25, "HN", ga_path(artifact_root, role, seed, 25)),
            ("ga50", 50, "HN", ga_path(artifact_root, role, seed, 50)),
            ("ga75", 75, "HN", ga_path(artifact_root, role, seed, 75)),
            ("ga100", 100, "EXPERIMENTAL", ga_path(artifact_root, role, seed, 100)),
        ]
        for condition, fraction, hypothesis, checkpoint in entries:
            rows.append({
                "model_id": f"{role}_seed{seed}_{condition}",
                "role": role,
                "seed": seed,
                "condition": condition,
                "fraction": fraction,
                "hypothesis": hypothesis,
                "checkpoint": str(checkpoint),
                "exists": checkpoint.is_file(),
            })
    return pd.DataFrame(rows)
