from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

STAGE2_DIR = Path(__file__).resolve().parents[1]
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from prismu_stage2.core import (
    blackbox_probabilities,
    load_model,
    load_split,
    load_yaml,
    make_datasets,
    make_loader,
    validate_split,
)
from prismu_stage2.layout import build_model_manifest, resolved_paths


def parse_args():
    p = argparse.ArgumentParser(description="Freeze the Stage 2 query pool and cache black-box responses.")
    p.add_argument("--config", default="stage2/configs/stage2.yaml")
    p.add_argument("--role", choices=["all", "surrogate", "validation", "final"], default="all")
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path.cwd().resolve()
    cfg = load_yaml(Path(args.config))
    paths = resolved_paths(cfg, repo_root)
    artifact_root = paths["artifact_root"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = int(cfg["runtime"]["num_workers"])
    batch_size = int(cfg["runtime"]["eval_batch_size"])
    num_classes = int(cfg["model"]["num_classes"])

    split = load_split(paths["stage1_split"])
    validate_split(split, {"auxiliary": int(cfg["query_pool"]["expected_size"])})
    _train_aug, train_eval, _test = make_datasets(paths["data_root"])

    query_indices = np.asarray(split[cfg["query_pool"]["split_name"]], dtype=np.int64)
    query_root = artifact_root / "query_pool"
    query_root.mkdir(parents=True, exist_ok=True)
    np.save(query_root / "query_indices.npy", query_indices)

    targets = np.asarray(train_eval.targets, dtype=np.int64)
    query_manifest = pd.DataFrame({
        "query_id": np.arange(len(query_indices), dtype=np.int64),
        "dataset_index": query_indices,
        "true_class": targets[query_indices],
    })
    query_manifest.to_csv(query_root / "query_manifest.csv", index=False)

    query_loader = make_loader(
        Subset(train_eval, query_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    manifest = build_model_manifest(cfg, repo_root)
    manifest_dir = artifact_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_dir / "model_manifest.csv", index=False)

    roles = ["surrogate", "validation", "final"] if args.role == "all" else [args.role]
    selected_manifest = manifest[manifest["role"].isin(roles)]
    missing = selected_manifest[~selected_manifest["exists"]]
    if not missing.empty:
        example = "\n".join(missing[["model_id", "checkpoint"]].head(10).astype(str).agg(" -> ".join, axis=1))
        raise FileNotFoundError(
            "Selected Stage 2 model bank is incomplete. Run build_bank.py first. Missing examples:\n" + example
        )

    response_root = artifact_root / "responses"
    response_root.mkdir(parents=True, exist_ok=True)

    for role in roles:
        role_df = manifest[manifest["role"] == role].reset_index(drop=True)
        cache_path = response_root / f"{role}_responses.npz"
        if cache_path.exists() and not args.force:
            print(f"[skip] response cache exists: {cache_path}")
            continue

        tensors = []
        for i, row in role_df.iterrows():
            print(f"[{role}] {i + 1}/{len(role_df)} querying {row['model_id']}")
            model = load_model(Path(row["checkpoint"]), device, num_classes)
            probs = blackbox_probabilities(model, query_loader, device)
            if probs.shape != (len(query_indices), num_classes):
                raise RuntimeError(f"Unexpected response shape for {row['model_id']}: {probs.shape}")
            if not np.isfinite(probs).all():
                raise RuntimeError(f"Non-finite probabilities from {row['model_id']}")
            if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
                raise RuntimeError(f"Probabilities do not sum to one for {row['model_id']}")
            tensors.append(probs)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        responses = np.stack(tensors, axis=0).astype(np.float32, copy=False)
        np.savez_compressed(
            cache_path,
            probabilities=responses,
            model_ids=role_df["model_id"].to_numpy(dtype=str),
            hypotheses=role_df["hypothesis"].to_numpy(dtype=str),
            conditions=role_df["condition"].to_numpy(dtype=str),
            seeds=role_df["seed"].to_numpy(dtype=np.int64),
            query_indices=query_indices,
        )
        print(f"Saved {cache_path}: {responses.shape}")

    print(f"Frozen query pool: {query_root / 'query_indices.npy'} ({len(query_indices)} queries)")


if __name__ == "__main__":
    main()
