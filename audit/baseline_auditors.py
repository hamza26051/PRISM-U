"""Online audit runner for Milestone 6 baseline policies (Random, Fixed, Privacy-Blind Active)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from audit.evidence import EvidenceState, SurrogateEvidenceModel


@dataclass
class StepRecord:
    """Single step diagnostic during an online black-box audit."""

    step: int
    query_id: int
    llr: float
    p_compliant: float


@dataclass
class BaselineAuditResult:
    """Complete trajectory returned by a fixed-budget baseline audit."""

    policy_name: str
    target_id: str
    query_budget: int
    steps: list[StepRecord]
    final_llr: float
    final_p_compliant: float

    def state_at_budget(self, b: int) -> StepRecord:
        if b <= 0 or b > len(self.steps):
            raise ValueError(f"Budget {b} out of trajectory bounds [1, {len(self.steps)}]")
        return self.steps[b - 1]


def run_baseline_audit(
    *,
    target: Any,
    policy: Any,
    evidence_model: SurrogateEvidenceModel,
    max_queries: int | None = None,
    candidate_ids: Sequence[int] | None = None,
) -> BaselineAuditResult:
    """Execute an online black-box audit up to max_queries using only target.query()."""
    if hasattr(policy, "reset"):
        policy.reset()

    budget = int(target.query_budget if max_queries is None else min(max_queries, target.query_budget))
    state: EvidenceState = evidence_model.initial_state()
    steps: list[StepRecord] = []

    for step_idx in range(1, budget + 1):
        # Pass current state if policy supports response-adaptive scoring
        if hasattr(policy, "select_next"):
            import inspect
            sig = inspect.signature(policy.select_next)
            kwargs = {}
            if "state" in sig.parameters:
                kwargs["state"] = state
            if "candidate_ids" in sig.parameters:
                kwargs["candidate_ids"] = candidate_ids
            qid = policy.select_next(target.transcript, **kwargs)
        else:
            raise RuntimeError(f"Policy {type(policy)} does not implement select_next().")

        response = target.query(qid)
        state = evidence_model.update(state, int(qid), response)
        steps.append(
            StepRecord(
                step=step_idx,
                query_id=int(qid),
                llr=float(state.llr),
                p_compliant=float(state.p_compliant),
            )
        )

    return BaselineAuditResult(
        policy_name=getattr(policy, "name", type(policy).__name__),
        target_id=str(target.target_id),
        query_budget=budget,
        steps=steps,
        final_llr=float(state.llr),
        final_p_compliant=float(state.p_compliant),
    )

