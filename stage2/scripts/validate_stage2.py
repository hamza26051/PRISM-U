from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGE2_DIR = Path(__file__).resolve().parents[1]
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from prismu_stage2.core import load_split, load_yaml, save_json, validate_split
from prismu_stage2.layout import build_model_manifest, resolved_paths


def parse_args():
    p = argparse.ArgumentParser(description="Run the hard completion gate for PRISM-U Stage 2.")
    p.add_argument("--config", default="stage2/configs/stage2.yaml")
    return p.parse_args()


def check_cache(path: Path, expected_models: int, expected_queries: int, num_classes: int) -> dict:
    if not path.is_file():
        return {"exists": False, "ok": False, "reason": "missing"}
    data = np.load(path, allow_pickle=False)
    probs = data["probabilities"]
    ok_shape = probs.shape == (expected_models, expected_queries, num_classes)
    finite = bool(np.isfinite(probs).all())
    simplex = bool(np.allclose(probs.sum(axis=2), 1.0, atol=1e-5)) if probs.ndim == 3 else False
    return {
        "exists": True,
        "shape": list(probs.shape),
        "expected_shape": [expected_models, expected_queries, num_classes],
        "shape_ok": bool(ok_shape),
        "finite": finite,
        "probability_simplex": simplex,
        "ok": bool(ok_shape and finite and simplex),
    }


def main():
    args = parse_args()
    repo_root = Path.cwd().resolve()
    cfg = load_yaml(Path(args.config))
    paths = resolved_paths(cfg, repo_root)
    artifact_root = paths["artifact_root"]
    expected_queries = int(cfg["query_pool"]["expected_size"])
    num_classes = int(cfg["model"]["num_classes"])

    split = load_split(paths["stage1_split"])
    split_report = validate_split(split, {"target_train": 40000, "auxiliary": expected_queries, "validation": 5000})

    manifest = build_model_manifest(cfg, repo_root)
    expected_counts = {
        "surrogate": 6 * len(cfg["bank"]["surrogate_seeds"]),
        "validation": 6 * len(cfg["bank"]["validation_seeds"]),
        "final": 6,
    }
    model_checks = {}
    for role, expected in expected_counts.items():
        role_df = manifest[manifest["role"] == role]
        model_checks[role] = {
            "expected": int(expected),
            "manifest_rows": int(len(role_df)),
            "checkpoints_present": int(role_df["exists"].sum()),
            "ok": bool(len(role_df) == expected and role_df["exists"].all()),
        }

    surrogate = manifest[manifest["role"] == "surrogate"]
    validation = manifest[manifest["role"] == "validation"]
    final = manifest[manifest["role"] == "final"]
    seed_sets = {
        "surrogate": set(surrogate["seed"].astype(int)),
        "validation": set(validation["seed"].astype(int)),
        "final": set(final["seed"].astype(int)),
    }
    seed_isolation = (
        seed_sets["surrogate"].isdisjoint(seed_sets["validation"])
        and seed_sets["surrogate"].isdisjoint(seed_sets["final"])
        and seed_sets["validation"].isdisjoint(seed_sets["final"])
    )

    label_checks = {
        "surrogate_hc": int((surrogate["hypothesis"] == "HC").sum()),
        "surrogate_hn": int((surrogate["hypothesis"] == "HN").sum()),
        "surrogate_experimental": int((surrogate["hypothesis"] == "EXPERIMENTAL").sum()),
        "validation_hc": int((validation["hypothesis"] == "HC").sum()),
        "validation_hn": int((validation["hypothesis"] == "HN").sum()),
    }
    labels_ok = (
        label_checks["surrogate_hc"] == len(cfg["bank"]["surrogate_seeds"])
        and label_checks["surrogate_hn"] == 4 * len(cfg["bank"]["surrogate_seeds"])
        and label_checks["surrogate_experimental"] == len(cfg["bank"]["surrogate_seeds"])
        and label_checks["validation_hc"] == len(cfg["bank"]["validation_seeds"])
        and label_checks["validation_hn"] == 4 * len(cfg["bank"]["validation_seeds"])
    )

    response_root = artifact_root / "responses"
    cache_checks = {
        role: check_cache(response_root / f"{role}_responses.npz", expected_counts[role], expected_queries, num_classes)
        for role in ["surrogate", "validation", "final"]
    }

    query_indices_path = artifact_root / "query_pool" / "query_indices.npy"
    query_manifest_path = artifact_root / "query_pool" / "query_manifest.csv"
    query_ok = False
    query_info = {"indices_exists": query_indices_path.is_file(), "manifest_exists": query_manifest_path.is_file()}
    if query_indices_path.is_file() and query_manifest_path.is_file():
        query_indices = np.load(query_indices_path)
        query_manifest = pd.read_csv(query_manifest_path)
        query_ok = (
            len(query_indices) == expected_queries
            and len(query_manifest) == expected_queries
            and set(map(int, query_indices)) == set(map(int, split["auxiliary"]))
            and set(map(int, query_indices)).isdisjoint(set(map(int, split["target_train"])))
            and set(map(int, query_indices)).isdisjoint(set(map(int, split["validation"])))
        )
        query_info.update({"size": int(len(query_indices)), "matches_auxiliary_exactly": set(map(int, query_indices)) == set(map(int, split["auxiliary"]))})
    query_info["ok"] = bool(query_ok)

    diag_root = artifact_root / "diagnostics"
    diagnostic_files = [
        diag_root / "query_separability.csv",
        diag_root / "validation_query_separability.csv",
        diag_root / "surrogate_diagnostics.json",
        diag_root / "validation_diagnostics.json",
        diag_root / "separability_histogram.png",
        diag_root / "top100_queries.png",
    ]
    diagnostics_ok = all(p.is_file() for p in diagnostic_files)

    checks = {
        "split_valid": True,
        "model_bank": model_checks,
        "seed_isolation": bool(seed_isolation),
        "labels": {**label_checks, "ok": bool(labels_ok)},
        "query_pool": query_info,
        "response_caches": cache_checks,
        "diagnostics_present": bool(diagnostics_ok),
        "final_seed_not_used_in_reference_banks": bool(int(cfg["bank"]["final_seed"]) not in seed_sets["surrogate"] | seed_sets["validation"]),
    }

    complete = (
        all(x["ok"] for x in model_checks.values())
        and seed_isolation
        and labels_ok
        and query_ok
        and all(x["ok"] for x in cache_checks.values())
        and diagnostics_ok
        and checks["final_seed_not_used_in_reference_banks"]
    )

    report = {
        "stage": 2,
        "stage2_complete": bool(complete),
        "scientific_boundary": "Stage 2 builds reference/query/response infrastructure. Privacy-aware acquisition, sequential LLR stopping, and PASS/FAIL/INCONCLUSIVE decisions begin in later stages.",
        "checks": checks,
    }
    report_path = artifact_root / "validation_report.json"
    save_json(report, report_path)
    print(f"Stage 2 complete: {complete}")
    print(f"Validation report: {report_path}")
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
