"""I/O utilities: YAML loading, JSON saving, path resolution, checkpoint validation."""
from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: Path) -> None:
    """Save a dict as indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (Path(repo_root) / p).resolve()


def assert_real_checkpoint(path: Path) -> None:
    """Verify a checkpoint file exists and is not a Git LFS pointer."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.stat().st_size < 1024:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "git-lfs.github.com/spec" in text:
            raise RuntimeError(f"{path} is a Git LFS pointer. Run `git lfs pull` first.")


def checkpoint_state_dict(checkpoint: object) -> dict:
    """Extract a model state dict from various checkpoint formats."""
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and checkpoint and all(isinstance(k, str) for k in checkpoint):
        return checkpoint
    raise ValueError("Unsupported checkpoint schema; expected model_state/state_dict.")
