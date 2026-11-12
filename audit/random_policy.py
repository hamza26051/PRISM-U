"""Random query baseline policy for PRISM-U."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from audit.transcript import AuditTranscript


class RandomQueryPolicy:
    """Uniformly samples unqueried candidates from the permitted auxiliary pool Q."""

    name = "random"

    def __init__(self, pool_size: int, seed: int = 42) -> None:
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0")
        self.pool_size = int(pool_size)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        """Reset RNG state for a fresh audit run."""
        self._rng = np.random.default_rng(self.seed)

    def select_next(self, transcript: AuditTranscript, candidate_ids: Sequence[int] | None = None) -> int:
        """Select one unqueried candidate uniformly at random."""
        queried = transcript.queried_id_set
        pool = range(self.pool_size) if candidate_ids is None else candidate_ids
        available = [int(q) for q in pool if int(q) not in queried]
        if not available:
            raise RuntimeError("No unqueried candidates remain in pool.")
        idx = int(self._rng.integers(0, len(available)))
        return available[idx]

