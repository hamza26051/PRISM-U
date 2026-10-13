from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

STAGE2_DIR = Path(__file__).resolve().parents[1]
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from prismu_stage2.core import (
    assert_real_checkpoint,
    evaluate,
    generate_gradient_ascent_family,
    load_model,
    load_split,
    load_yaml,
    make_datasets,
    make_loader,
    save_json,
    train_from_scratch,
    validate_split,
)
from prismu_stage2.layout import (
    build_model_manifest,
    family_dir,
    ga_path,
    iter_families,
    original_path,
    resolved_paths,
    retrain_path,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build the complete PRISM-U Stage 2 reference bank.")
    p.add_argument("--config", default="stage2/configs/stage2.yaml")
    p.add_argument("--role", choices=["all", "surrogate", "validation", "final"], default="all")
    p.add_argument("--seed", type=int, default=None, help="Optional single seed within the selected role.")
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def selected_families(cfg: dict, role: str, seed: int | None):
    families = list(iter_families(cfg))
    if role != "all":
        families = [x for x in families if x[0] == role]
    if seed is not None:
        families = [x for x in families if x[1] == seed]
    if not families:
        raise ValueError("No families match --role/--seed selection.")
    return families


def evaluate_family(*, role, seed, cfg, artifact_root, paths, final_seed, device, num_classes, retain_loader, forget_loader, validation_loader, test_loader, log_dir):
    checkpoints = {
        "original": original_path(artifact_root, role, seed, paths["stage1_original"], final_seed),
        "retrain": retrain_path(artifact_root, role, seed, paths["stage1_retrain"], final_seed),
    }
    for fraction in cfg["unlearning"]["fractions"]:
        checkpoints[f"ga{int(fraction)}"] = ga_path(artifact_root, role, seed, int(fraction))

    metrics = {"role": role, "seed": int(seed), "conditions": {}}
    for condition, checkpoint in checkpoints.items():
        model = load_model(checkpoint, device, num_classes)
        metrics["conditions"][condition] = {
            "checkpoint": str(checkpoint),
            "retain": evaluate(model, retain_loader, device),
            "forget": evaluate(model, forget_loader, device),
            "validation": evaluate(model, validation_loader, device),
            "test": evaluate(model, test_loader, device),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(metrics, log_dir / "family_evaluation.json")


def main():
    args = parse_args()
    repo_root = Path.cwd().resolve()
    cfg = load_yaml(Path(args.config))
    paths = resolved_paths(cfg, repo_root)
    artifact_root = paths["artifact_root"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = int(cfg["runtime"]["num_workers"])
    eval_batch = int(cfg["runtime"]["eval_batch_size"])
    num_classes = int(cfg["model"]["num_classes"])

    split = load_split(paths["stage1_split"])
    split_report = validate_split(
        split,
        expected_sizes={
            "target_train": 40000,
            "auxiliary": int(cfg["query_pool"]["expected_size"]),
            "validation": 5000,
        },
    )
    save_json(split_report, artifact_root / "reports" / "split_validation.json")

    # Seed 42 comes from Stage 1. Catch missing Git LFS bytes before expensive work.
    assert_real_checkpoint(paths["stage1_original"])
    assert_real_checkpoint(paths["stage1_retrain"])

    train_aug, train_eval, test_data = make_datasets(paths["data_root"])
    validation_loader = make_loader(
        Subset(train_eval, split["validation"]), eval_batch, False, num_workers, pin_memory=device.startswith("cuda")
    )
    retain_loader = make_loader(
        Subset(train_eval, split["retain"]), eval_batch, False, num_workers, pin_memory=device.startswith("cuda")
    )
    forget_loader = make_loader(
        Subset(train_eval, split["forget"]), eval_batch, False, num_workers, pin_memory=device.startswith("cuda")
    )
    test_loader = make_loader(test_data, eval_batch, False, num_workers, pin_memory=device.startswith("cuda"))
    forget_eval_dataset = Subset(train_eval, split["forget"])

    final_seed = int(cfg["bank"]["final_seed"])
    for role, seed in selected_families(cfg, args.role, args.seed):
        print("\n" + "=" * 72)
        print(f"Building role={role} seed={seed} on {device}")
        print("=" * 72)
        out_original = original_path(artifact_root, role, seed, paths["stage1_original"], final_seed)
        out_retrain = retrain_path(artifact_root, role, seed, paths["stage1_retrain"], final_seed)
        log_dir = artifact_root / "logs" / role / f"seed{seed}"

        if role != "final":
            train_from_scratch(
                seed=seed,
                mode="original",
                indices=split["target_train"],
                train_aug_dataset=train_aug,
                validation_loader=validation_loader,
                retain_loader=retain_loader,
                forget_loader=forget_loader,
                test_loader=test_loader,
                output_path=out_original,
                log_path=log_dir / "train_original.json",
                device=device,
                num_classes=num_classes,
                training_cfg=cfg["training"],
                num_workers=num_workers,
                force=args.force,
            )
            train_from_scratch(
                seed=seed,
                mode="retrain",
                indices=split["retain"],
                train_aug_dataset=train_aug,
                validation_loader=validation_loader,
                retain_loader=retain_loader,
                forget_loader=forget_loader,
                test_loader=test_loader,
                output_path=out_retrain,
                log_path=log_dir / "train_retrain.json",
                device=device,
                num_classes=num_classes,
                training_cfg=cfg["training"],
                num_workers=num_workers,
                force=args.force,
            )

        generate_gradient_ascent_family(
            seed=seed,
            original_checkpoint=out_original,
            forget_eval_dataset=forget_eval_dataset,
            output_dir=family_dir(artifact_root, role, seed) / "ga",
            device=device,
            num_classes=num_classes,
            batch_size=int(cfg["unlearning"]["batch_size"]),
            num_workers=num_workers,
            total_steps=int(cfg["unlearning"]["total_steps"]),
            learning_rate=float(cfg["unlearning"]["learning_rate"]),
            fractions=cfg["unlearning"]["fractions"],
            force=args.force,
        )

        evaluate_family(
            role=role,
            seed=seed,
            cfg=cfg,
            artifact_root=artifact_root,
            paths=paths,
            final_seed=final_seed,
            device=device,
            num_classes=num_classes,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            log_dir=log_dir,
        )

        manifest = build_model_manifest(cfg, repo_root)
        manifest_dir = artifact_root / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_dir / "model_manifest.csv", index=False)

    manifest = build_model_manifest(cfg, repo_root)
    print("\nBank status:")
    print(manifest.groupby(["role", "hypothesis"])["exists"].agg(["sum", "count"]))
    print(f"Manifest: {artifact_root / 'manifests' / 'model_manifest.csv'}")


if __name__ == "__main__":
    main()
