"""Tests for Milestone 4: strict black-box target interface, isolation, budgets, and parity."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from audit.transcript import AuditTranscript
from models.black_box_target import (
    BlackBoxIsolationError,
    CachedBlackBoxTarget,
    InvalidQueryError,
    PyTorchBlackBoxTarget,
    QueryBudgetExhaustedError,
    build_auxiliary_query_dataset,
    load_cached_target_bank,
)
from models.resnet import build_resnet18_cifar

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_audit_modules_cannot_import_model_or_checkpoint_internals():
    """Milestone 4 exit check: verify audit/ files do not import weights, checkpoints, or torch.load."""
    forbidden_modules = {"models.resnet", "models.train", "unlearning", "torch"}
    forbidden_names = {"load_model", "torch.load", "state_dict", "checkpoint_state_dict", "__closure__"}

    audit_dir = ROOT_DIR / "audit"
    py_files = list(audit_dir.rglob("*.py"))
    assert py_files, "Expected Python files in audit/"

    for py_file in py_files:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_modules, (
                        f"{py_file.name} imports forbidden module {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                assert node.module not in forbidden_modules and top != "torch", (
                    f"{py_file.name} imports from forbidden module {node.module}"
                )
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden_names, (
                    f"{py_file.name} references forbidden symbol {node.id}"
                )
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_names, (
                    f"{py_file.name} accesses forbidden attribute {node.attr}"
                )


def test_pytorch_target_isolation_and_access_modes():
    model = build_resnet18_cifar(num_classes=10)
    target = PyTorchBlackBoxTarget.from_model(
        model, device="cpu", target_id="synthetic_001", num_classes=10, query_budget=4, access_mode="probabilities"
    )

    for attr in ("model", "_model", "weights", "parameters", "state_dict", "grad", "checkpoint", "hypothesis", "condition"):
        with pytest.raises(BlackBoxIsolationError):
            getattr(target, attr)
        assert attr not in vars(target)

    x = torch.randn(2, 3, 32, 32)
    probs = target.query_tensor(x)
    assert probs.shape == (2, 10)
    assert probs.dtype == np.float32
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert target.queries_used == 2
    assert target.queries_remaining == 2

    # Mutating returned array must not corrupt transcript
    probs_copy = probs.copy()
    probs[0, :] = -999.0
    assert np.allclose(target.transcript.responses, probs_copy)

    # Overshoot batch must raise cleanly without incrementing queries_used
    with pytest.raises(QueryBudgetExhaustedError):
        target.query_tensor(torch.randn(3, 3, 32, 32))
    assert target.queries_used == 2
    assert target.queries_remaining == 2

    # Logits and top1 modes
    logits_target = PyTorchBlackBoxTarget.from_model(
        model, device="cpu", target_id="t_logits", num_classes=10, query_budget=2, access_mode="logits"
    )
    logits = logits_target.query_tensor(x)
    assert logits.shape == (2, 10)
    assert not np.allclose(logits.sum(axis=1), 1.0, atol=1e-3)

    top1_target = PyTorchBlackBoxTarget.from_model(
        model, device="cpu", target_id="t_top1", num_classes=10, query_budget=2, access_mode="top1"
    )
    preds = top1_target.query_tensor(x)
    assert preds.shape == (2,)
    assert np.issubdtype(preds.dtype, np.integer)
    assert np.array_equal(preds, logits.argmax(axis=1))


def test_cached_target_budget_pool_and_transcript(tmp_path):
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
        [0.9, 0.05, 0.05],
    ], dtype=np.float32)
    q_indices = [100, 200, 300, 400]

    target = CachedBlackBoxTarget(
        prob_matrix=probs,
        target_id="cached_001",
        query_budget=3,
        query_indices=q_indices,
        access_mode="probabilities",
    )

    with pytest.raises(BlackBoxIsolationError):
        _ = target._probs
    with pytest.raises(BlackBoxIsolationError):
        target.query_tensor(torch.randn(1, 3, 32, 32))
    with pytest.raises(InvalidQueryError):
        target.query(4)  # out of pool range [0, 3]
    with pytest.raises(InvalidQueryError):
        target.query(-1)
    assert target.queries_used == 0

    r0 = target.query(2)
    assert np.allclose(r0, probs[2])
    r_batch = target.query_batch([0, 3])
    assert r_batch.shape == (2, 3)
    assert np.allclose(r_batch[0], probs[0])
    assert np.allclose(r_batch[1], probs[3])
    assert target.queries_used == 3
    assert target.queries_remaining == 0

    with pytest.raises(QueryBudgetExhaustedError):
        target.query(1)

    out_json = tmp_path / "transcript.json"
    target.transcript.save_json(out_json)
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["queries_used"] == 3
    assert [e["query_id"] for e in saved["entries"]] == [2, 0, 3]
    assert [e["dataset_index"] for e in saved["entries"]] == [300, 100, 400]

    # Top-1 cached mode and logits rejection
    top1_target = CachedBlackBoxTarget(
        prob_matrix=probs, target_id="t1", query_budget=2, query_indices=q_indices, access_mode="top1"
    )
    assert top1_target.query(2) == 2
    assert isinstance(top1_target.query(1), int)

    with pytest.raises(ValueError):
        CachedBlackBoxTarget(
            prob_matrix=probs, target_id="t_log", query_budget=2, query_indices=q_indices, access_mode="logits"
        )


def test_live_pytorch_vs_cached_target_equivalence_on_real_validation_bank():
    """Verify live PyTorch checkpoint inference matches Stage 2 cached black-box responses across target types."""
    val_cache = ROOT_DIR / "stage2_artifacts" / "responses" / "validation_responses.npz"
    split_file = ROOT_DIR / "splits" / "cifar10_seed42.json"
    data_root = ROOT_DIR / "data" / "cifar10"
    if not (val_cache.is_file() and split_file.is_file() and data_root.is_dir()):
        pytest.skip("Stage 2 artifacts or CIFAR-10 data not present.")

    aux_ds, q_indices = build_auxiliary_query_dataset(data_root, split_file, "auxiliary")
    bank = load_cached_target_bank(val_cache, query_budget=10, access_mode="probabilities", anonymize_ids=True)

    # Test equivalence on three distinct target types from validation seed 201: retrain (HC), original (HN), ga50 (HN)
    target_cases = [
        ("validation_seed201_retrain", ROOT_DIR / "stage2_artifacts" / "checkpoints" / "validation" / "seed201" / "retrain.pt"),
        ("validation_seed201_original", ROOT_DIR / "stage2_artifacts" / "checkpoints" / "validation" / "seed201" / "original.pt"),
        ("validation_seed201_ga50", ROOT_DIR / "stage2_artifacts" / "checkpoints" / "validation" / "seed201" / "ga" / "ga50.pt"),
    ]
    probe_ids = [0, 17, 123, 999, 2500, 4999]

    by_model_id = {item["ground_truth"]["model_id"]: item for item in bank}
    for model_id, ckpt_path in target_cases:
        item = by_model_id[model_id]
        cached_target = item["target"]
        assert cached_target.target_id.startswith("target_")  # anonymized!

        live_target = PyTorchBlackBoxTarget.from_checkpoint(
            ckpt_path,
            device="cpu",
            target_id=cached_target.target_id,
            num_classes=10,
            query_budget=10,
            access_mode="probabilities",
            query_dataset=aux_ds,
            query_indices=q_indices,
        )
        # Pool-bound live target must forbid arbitrary tensor probing
        with pytest.raises(BlackBoxIsolationError):
            live_target.query_tensor(torch.randn(1, 3, 32, 32))

        cached_res = cached_target.query_batch(probe_ids)
        live_res = live_target.query_batch(probe_ids)
        assert cached_res.shape == (len(probe_ids), 10)
        # Float32 convolution reduction order across CPU vs CUDA/batch sizes can differ by ~3e-4.
        assert np.allclose(live_res, cached_res, atol=1e-3)
        assert np.array_equal(live_res.argmax(axis=1), cached_res.argmax(axis=1))
        assert live_target.transcript.query_ids == probe_ids
        assert cached_target.transcript.query_ids == probe_ids
