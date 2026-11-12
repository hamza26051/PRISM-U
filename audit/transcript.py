"""Append-only black-box audit transcript container and serialization utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TranscriptEntry:
    """Immutable record of a single black-box query-response interaction."""

    step: int
    query_id: Optional[int]
    dataset_index: Optional[int]
    response: np.ndarray
    access_mode: str
    timestamp_utc: str

    def __post_init__(self) -> None:
        arr = np.array(self.response, copy=True)
        arr.setflags(write=False)
        object.__setattr__(self, "response", arr)

    def to_dict(self) -> dict:
        """Convert entry to JSON-serializable dictionary."""
        resp_val: object
        if self.response.ndim == 0:
            resp_val = int(self.response.item()) if np.issubdtype(self.response.dtype, np.integer) else float(self.response.item())
        else:
            resp_val = self.response.tolist()
        return {
            "step": int(self.step),
            "query_id": None if self.query_id is None else int(self.query_id),
            "dataset_index": None if self.dataset_index is None else int(self.dataset_index),
            "response": resp_val,
            "access_mode": str(self.access_mode),
            "timestamp_utc": str(self.timestamp_utc),
        }


class AuditTranscript:
    """Append-only sequence tau_t = {(x_1, r_1), ..., (x_t, r_t)} for one audit."""

    def __init__(self, target_id: str, access_mode: str, query_budget: int) -> None:
        self._target_id = str(target_id)
        self._access_mode = str(access_mode)
        self._query_budget = int(query_budget)
        self._entries: list[TranscriptEntry] = []

    @property
    def target_id(self) -> str:
        return self._target_id

    @property
    def access_mode(self) -> str:
        return self._access_mode

    @property
    def query_budget(self) -> int:
        return self._query_budget

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[TranscriptEntry, ...]:
        """Return an immutable snapshot tuple of all transcript entries."""
        return tuple(self._entries)

    @property
    def query_ids(self) -> list[int]:
        """Return ordered query_ids recorded in the transcript (excluding None)."""
        return [e.query_id for e in self._entries if e.query_id is not None]

    @property
    def queried_id_set(self) -> set[int]:
        """Return set of distinct query_ids already queried."""
        return {e.query_id for e in self._entries if e.query_id is not None}

    @property
    def responses(self) -> np.ndarray:
        """Return a defensive copy of stacked response arrays of shape [t, ...]."""
        if not self._entries:
            return np.empty((0,), dtype=np.float32)
        return np.stack([e.response for e in self._entries], axis=0).copy()

    def append(
        self,
        *,
        query_id: Optional[int],
        dataset_index: Optional[int],
        response: np.ndarray,
        access_mode: Optional[str] = None,
    ) -> TranscriptEntry:
        """Append one query-response observation and return the immutable entry."""
        mode = self._access_mode if access_mode is None else str(access_mode)
        if mode != self._access_mode:
            raise ValueError(f"Transcript access_mode mismatch: expected {self._access_mode}, got {mode}")
        entry = TranscriptEntry(
            step=len(self._entries) + 1,
            query_id=None if query_id is None else int(query_id),
            dataset_index=None if dataset_index is None else int(dataset_index),
            response=np.asarray(response),
            access_mode=mode,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._entries.append(entry)
        return entry

    def to_dict(self) -> dict:
        """Export full transcript metadata and entries as JSON-ready dictionary."""
        return {
            "target_id": self._target_id,
            "access_mode": self._access_mode,
            "query_budget": self._query_budget,
            "queries_used": len(self._entries),
            "unique_queries": len(self.queried_id_set),
            "entries": [e.to_dict() for e in self._entries],
        }

    def save_json(self, path: Path) -> None:
        """Save transcript to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

