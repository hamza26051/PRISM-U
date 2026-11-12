"""Strict black-box target prediction interface for PRISM-U auditing.

Enforces:
- Output-only access (probabilities, logits, or top-1 predictions)
- Query budget limits (B_Q) with atomic pre-flight checks
- Permitted auxiliary pool enforcement (prevents probing D_u or D_r)
- Append-only audit transcript logging (tau_t)
- Complete isolation of target weights, gradients, checkpoints, and ground-truth labels
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from audit.transcript import AuditTranscript
from data.datasets import make_datasets
from data.splits import load_split
from models.resnet import load_model

VALID_ACCESS_MODES = ("probabilities", "logits", "top1")

_FORBIDDEN_ATTRS = frozenset({
    "model",
    "_model",
    "weights",
    "parameters",
    "named_parameters",
    "buffers",
    "state_dict",
    "grad",
    "gradients",
    "optimizer",
    "checkpoint",
    "checkpoint_path",
    "hypothesis",
    "condition",
    "true_label",
    "seed",
    "role",
    "training_data",
    "_probs",
    "_responses",
})


class QueryBudgetExhaustedError(RuntimeError):
    """Raised when a query or batch would exceed the target's query budget B_Q."""


class InvalidQueryError(ValueError):
    """Raised when a query ID or input violates permitted pool or format constraints."""


class BlackBoxIsolationError(PermissionError):
    """Raised when code attempts to access forbidden target internals or metadata."""


class BlackBoxTarget(ABC):
    """Abstract black-box oracle interface exposed to PRISM-U auditors."""

    def __init__(
        self,
        *,
        target_id: str,
        num_classes: int,
        query_budget: int,
        access_mode: str = "probabilities",
        query_indices: Optional[Sequence[int]] = None,
        allow_arbitrary_tensors: bool = False,
    ) -> None:
        if access_mode not in VALID_ACCESS_MODES:
            raise ValueError(f"Unsupported access_mode '{access_mode}'; expected one of {VALID_ACCESS_MODES}")
        if int(query_budget) < 0:
            raise ValueError(f"query_budget must be >= 0, got {query_budget}")
        if int(num_classes) <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")

        self._target_id = str(target_id)
        self._num_classes = int(num_classes)
        self._query_budget = int(query_budget)
        self._access_mode = str(access_mode)
        self._allow_arbitrary_tensors = bool(allow_arbitrary_tensors)
        self._query_indices: Optional[tuple[int, ...]] = (
            None if query_indices is None else tuple(int(x) for x in query_indices)
        )
        self._transcript = AuditTranscript(
            target_id=self._target_id,
            access_mode=self._access_mode,
            query_budget=self._query_budget,
        )

    def __getattr__(self, name: str):
        if name in _FORBIDDEN_ATTRS:
            raise BlackBoxIsolationError(
                f"Access to '{name}' is forbidden: auditor must interact solely via black-box queries."
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def target_id(self) -> str:
        return self._target_id

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def access_mode(self) -> str:
        return self._access_mode

    @property
    def query_budget(self) -> int:
        return self._query_budget

    @property
    def queries_used(self) -> int:
        return len(self._transcript)

    @property
    def queries_remaining(self) -> int:
        return self._query_budget - len(self._transcript)

    @property
    def pool_size(self) -> Optional[int]:
        return None if self._query_indices is None else len(self._query_indices)

    @property
    def transcript(self) -> AuditTranscript:
        return self._transcript

    def _check_budget(self, count: int) -> None:
        if count <= 0:
            raise InvalidQueryError("Query batch must contain at least 1 query.")
        if self.queries_used + count > self._query_budget:
            raise QueryBudgetExhaustedError(
                f"Query request of size {count} exceeds budget B_Q={self._query_budget} "
                f"(used={self.queries_used}, remaining={self.queries_remaining})."
            )

    def _validate_pool_ids(self, query_ids: Sequence[int]) -> list[int]:
        if self._query_indices is None:
            raise InvalidQueryError("This target is not configured with a discrete query pool; use query_tensor().")
        ids = [int(qid) for qid in query_ids]
        n = len(self._query_indices)
        bad = [qid for qid in ids if qid < 0 or qid >= n]
        if bad:
            raise InvalidQueryError(f"Query IDs out of permitted pool range [0, {n - 1}]: {bad[:5]}")
        return ids

    @abstractmethod
    def _infer_pool_batch(self, query_ids: Sequence[int]) -> np.ndarray:
        """Run black-box inference on a validated list of pool query_ids."""

    def query_batch(self, query_ids: Sequence[int]) -> np.ndarray:
        """Query a batch of permitted pool indices, update transcript, and return responses."""
        ids = list(query_ids)
        self._check_budget(len(ids))
        validated_ids = self._validate_pool_ids(ids)
        responses = self._infer_pool_batch(validated_ids)

        assert self._query_indices is not None
        for idx, qid in enumerate(validated_ids):
            self._transcript.append(
                query_id=qid,
                dataset_index=self._query_indices[qid],
                response=responses[idx],
                access_mode=self._access_mode,
            )
        return np.array(responses, copy=True)

    def query(self, query_id: int) -> np.ndarray | int:
        """Query a single permitted pool index and return its response."""
        out = self.query_batch([int(query_id)])
        if self._access_mode == "top1":
            return int(out[0])
        return out[0]


def _format_logits(logits: torch.Tensor, access_mode: str) -> np.ndarray:
    if access_mode == "probabilities":
        arr = F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32, copy=True)
        return arr
    if access_mode == "logits":
        return logits.detach().cpu().numpy().astype(np.float32, copy=True)
    if access_mode == "top1":
        return logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64, copy=True)
    raise ValueError(f"Unsupported access_mode: {access_mode}")


def _build_pytorch_closure(model: nn.Module, device: str, access_mode: str) -> Callable[[torch.Tensor], np.ndarray]:
    """Capture a PyTorch model inside a non-inspectable inference closure."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def _infer(images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            batch = images.to(device, non_blocking=False)
            logits = model(batch)
            return _format_logits(logits, access_mode)

    return _infer


class PyTorchBlackBoxTarget(BlackBoxTarget):
    """Black-box wrapper around a live PyTorch nn.Module via a closure-isolated oracle."""

    def __init__(
        self,
        *,
        infer_fn: Callable[[torch.Tensor], np.ndarray],
        target_id: str,
        num_classes: int,
        query_budget: int,
        access_mode: str = "probabilities",
        query_dataset: Optional[Dataset] = None,
        query_indices: Optional[Sequence[int]] = None,
        allow_arbitrary_tensors: bool = False,
        eval_batch_size: int = 256,
    ) -> None:
        super().__init__(
            target_id=target_id,
            num_classes=num_classes,
            query_budget=query_budget,
            access_mode=access_mode,
            query_indices=query_indices,
            allow_arbitrary_tensors=allow_arbitrary_tensors,
        )
        if query_dataset is not None and query_indices is not None:
            if len(query_dataset) != len(query_indices):
                raise ValueError(
                    f"query_dataset length ({len(query_dataset)}) must match query_indices ({len(query_indices)})."
                )
        self._infer_fn = infer_fn
        self._query_dataset = query_dataset
        self._eval_batch_size = int(eval_batch_size)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        device: str = "cpu",
        target_id: str = "target",
        num_classes: int = 10,
        query_budget: int = 100,
        access_mode: str = "probabilities",
        query_dataset: Optional[Dataset] = None,
        query_indices: Optional[Sequence[int]] = None,
        allow_arbitrary_tensors: Optional[bool] = None,
        eval_batch_size: int = 256,
    ) -> "PyTorchBlackBoxTarget":
        """Wrap an in-memory PyTorch model inside an isolated black-box oracle."""
        model = model.to(device)
        infer_fn = _build_pytorch_closure(model, device=device, access_mode=access_mode)
        if allow_arbitrary_tensors is None:
            allow_arbitrary_tensors = query_dataset is None
        return cls(
            infer_fn=infer_fn,
            target_id=target_id,
            num_classes=num_classes,
            query_budget=query_budget,
            access_mode=access_mode,
            query_dataset=query_dataset,
            query_indices=query_indices,
            allow_arbitrary_tensors=allow_arbitrary_tensors,
            eval_batch_size=eval_batch_size,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        device: str = "cpu",
        target_id: str = "target",
        num_classes: int = 10,
        query_budget: int = 100,
        access_mode: str = "probabilities",
        query_dataset: Optional[Dataset] = None,
        query_indices: Optional[Sequence[int]] = None,
        allow_arbitrary_tensors: Optional[bool] = None,
        eval_batch_size: int = 256,
    ) -> "PyTorchBlackBoxTarget":
        """Load a target checkpoint from disk and isolate it inside a black-box wrapper."""
        model = load_model(Path(checkpoint_path), device=device, num_classes=num_classes)
        return cls.from_model(
            model,
            device=device,
            target_id=target_id,
            num_classes=num_classes,
            query_budget=query_budget,
            access_mode=access_mode,
            query_dataset=query_dataset,
            query_indices=query_indices,
            allow_arbitrary_tensors=allow_arbitrary_tensors,
            eval_batch_size=eval_batch_size,
        )

    def _infer_pool_batch(self, query_ids: Sequence[int]) -> np.ndarray:
        if self._query_dataset is None:
            raise InvalidQueryError("No query_dataset configured on PyTorchBlackBoxTarget.")
        imgs = [self._query_dataset[qid][0] for qid in query_ids]
        tensor_batch = torch.stack(imgs, dim=0)
        chunks = []
        for start in range(0, len(tensor_batch), self._eval_batch_size):
            chunks.append(self._infer_fn(tensor_batch[start : start + self._eval_batch_size]))
        return np.concatenate(chunks, axis=0)

    def query_tensor(
        self,
        images: torch.Tensor,
        query_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Query arbitrary image tensors (only enabled when allow_arbitrary_tensors=True)."""
        if not self._allow_arbitrary_tensors:
            raise BlackBoxIsolationError(
                "Arbitrary tensor queries are disabled; this target only permits fixed auxiliary pool query_ids."
            )
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise InvalidQueryError(f"Expected 4D tensor [B, C, H, W] or 3D [C, H, W], got {tuple(images.shape)}")
        n = int(images.shape[0])
        self._check_budget(n)

        if query_ids is not None:
            if len(query_ids) != n:
                raise InvalidQueryError(f"query_ids length ({len(query_ids)}) != batch size ({n})")
            qids: list[Optional[int]] = [int(q) for q in query_ids]
        else:
            qids = [None] * n

        chunks = []
        for start in range(0, n, self._eval_batch_size):
            chunks.append(self._infer_fn(images[start : start + self._eval_batch_size]))
        responses = np.concatenate(chunks, axis=0)

        for idx in range(n):
            self._transcript.append(
                query_id=qids[idx],
                dataset_index=None,
                response=responses[idx],
                access_mode=self._access_mode,
            )
        return np.array(responses, copy=True)


class CachedBlackBoxTarget(BlackBoxTarget):
    """Fast, deterministic black-box wrapper backed by cached Stage 2 probability tensors."""

    def __init__(
        self,
        *,
        prob_matrix: np.ndarray,
        target_id: str,
        query_budget: int,
        query_indices: Sequence[int],
        access_mode: str = "probabilities",
    ) -> None:
        if access_mode == "logits":
            raise ValueError("CachedBlackBoxTarget stores softmax probabilities only; logits are unavailable.")
        probs = np.asarray(prob_matrix, dtype=np.float32)
        if probs.ndim != 2:
            raise ValueError(f"Expected 2D response array [|Q|, K], got {probs.shape}")
        if len(probs) != len(query_indices):
            raise ValueError(f"prob_matrix length ({len(probs)}) != query_indices ({len(query_indices)})")

        super().__init__(
            target_id=target_id,
            num_classes=int(probs.shape[1]),
            query_budget=query_budget,
            access_mode=access_mode,
            query_indices=query_indices,
            allow_arbitrary_tensors=False,
        )
        # Keep cached array privately closure-captured so instance __dict__ never holds raw table.
        cached_copy = np.array(probs, dtype=np.float32, copy=True)
        cached_copy.setflags(write=False)

        def _slice_fn(ids: Sequence[int]) -> np.ndarray:
            rows = cached_copy[list(ids)]
            if access_mode == "top1":
                return rows.argmax(axis=1).astype(np.int64, copy=True)
            return np.array(rows, dtype=np.float32, copy=True)

        self._slice_fn = _slice_fn

    def _infer_pool_batch(self, query_ids: Sequence[int]) -> np.ndarray:
        return self._slice_fn(query_ids)

    def query_tensor(self, *args, **kwargs):
        raise BlackBoxIsolationError("CachedBlackBoxTarget only supports permitted auxiliary pool query_ids.")


def build_auxiliary_query_dataset(data_root: Path, split_file: Path, split_name: str = "auxiliary") -> tuple[Dataset, list[int]]:
    """Load the frozen auxiliary query subset (eval transform) and its dataset indices."""
    split = load_split(split_file)
    query_indices = [int(x) for x in split[split_name]]
    _train_aug, train_eval, _test = make_datasets(data_root)
    return Subset(train_eval, query_indices), query_indices


def load_cached_target_bank(
    cache_path: Path,
    *,
    query_budget: int,
    access_mode: str = "probabilities",
    anonymize_ids: bool = True,
) -> list[dict]:
    """Load all targets from a Stage 2 .npz response cache.

    Returns a list of dicts separating `target` (the strict BlackBoxTarget passed to
    the auditor) from `ground_truth` (held strictly by the evaluation harness).
    """
    data = np.load(cache_path, allow_pickle=False)
    probs = data["probabilities"]
    model_ids = data["model_ids"].astype(str)
    hypotheses = data["hypotheses"].astype(str)
    conditions = data["conditions"].astype(str)
    seeds = data["seeds"].astype(np.int64)
    query_indices = data["query_indices"].astype(np.int64)

    items = []
    for i in range(len(probs)):
        public_id = f"target_{i:03d}" if anonymize_ids else str(model_ids[i])
        target = CachedBlackBoxTarget(
            prob_matrix=probs[i],
            target_id=public_id,
            query_budget=query_budget,
            query_indices=query_indices.tolist(),
            access_mode=access_mode,
        )
        items.append({
            "target": target,
            "ground_truth": {
                "model_id": str(model_ids[i]),
                "hypothesis": str(hypotheses[i]),
                "condition": str(conditions[i]),
                "seed": int(seeds[i]),
            },
        })
    return items

