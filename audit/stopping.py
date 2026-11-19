"""Budget-aware sequential stopping engine and tri-state decision protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from audit.evidence import EvidenceState, SurrogateEvidenceModel
from audit.prism_u import NoFeasibleQueriesError, PRISMUPolicy
from audit.transcript import AuditTranscript


@dataclass(frozen=True)
class AuditStepRecord:
    """Record of a single step during sequential auditing."""
    step: int
    query_id: int
    llr: float
    p_compliant: float
    cumulative_leakage: float


@dataclass
class AuditResult:
    """Outcome of a sequential compliance audit.

    Attributes:
        verdict: 'PASS', 'FAIL', or 'INCONCLUSIVE'.
        reason: Explanation of termination ('threshold_pass', 'threshold_fail',
                'query_budget_exhausted', 'privacy_budget_exhausted', 'no_feasible_queries').
        queries_used: Number of queries issued to the target.
        final_llr: Final log-likelihood ratio.
        final_p_compliant: Final posterior probability of compliance.
        cumulative_leakage: Final estimated privacy leakage L_hat(tau).
        target_id: Target identifier.
        policy_name: Policy name used for the audit.
        steps: Step-by-step diagnostic records.
        transcript: Complete immutable AuditTranscript.
    """
    verdict: str
    reason: str
    queries_used: int
    final_llr: float
    final_p_compliant: float
    cumulative_leakage: float
    target_id: str
    policy_name: str
    steps: list[AuditStepRecord]
    transcript: AuditTranscript

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "queries_used": self.queries_used,
            "final_llr": self.final_llr,
            "final_p_compliant": self.final_p_compliant,
            "cumulative_leakage": self.cumulative_leakage,
            "target_id": self.target_id,
            "policy_name": self.policy_name,
            "steps": [
                {
                    "step": s.step,
                    "query_id": s.query_id,
                    "llr": s.llr,
                    "p_compliant": s.p_compliant,
                    "cumulative_leakage": s.cumulative_leakage,
                }
                for s in self.steps
            ],
        }


@dataclass(frozen=True)
class StoppingConfig:
    """Configuration for sequential stopping decision boundaries and budget caps."""
    theta_pass: float = 2.944   # LLR >= theta_pass -> PASS (p_C >= 0.95)
    theta_fail: float = -2.944  # LLR <= theta_fail -> FAIL (p_C <= 0.05)
    query_budget: int = 160     # Hard query budget cap B_Q
    privacy_budget: float = float("inf")  # Hard privacy budget cap B_P
    min_gain_delta: float = 1e-4          # Minimum gain threshold for continuation
    force_max_budget: bool = False        # If True, disable early stopping (ablation)


class SequentialStoppingEngine:
    """Online sequential evidence accumulation and stopping decision engine."""

    def __init__(
        self,
        evidence_model: SurrogateEvidenceModel,
        config: Optional[StoppingConfig] = None,
    ) -> None:
        self.evidence_model = evidence_model
        self.config = config or StoppingConfig()

    def run_audit(
        self,
        *,
        target: Any,
        policy: Any,
        candidate_ids: Optional[Sequence[int]] = None,
        privacy_estimator: Optional[Any] = None,
    ) -> AuditResult:
        """Execute the online sequential audit loop against a BlackBoxTarget."""
        if hasattr(policy, "reset"):
            policy.reset()

        state: EvidenceState = self.evidence_model.initial_state()
        transcript = target.transcript
        steps: list[AuditStepRecord] = []
        max_b = min(self.config.query_budget, target.query_budget)
        bp = self.config.privacy_budget

        cur_leakage = 0.5
        policy_name = getattr(policy, "name", type(policy).__name__)

        while True:
            t = len(steps)

            # 1. Early stopping checks (unless disabled by ablation)
            if not self.config.force_max_budget:
                if state.llr >= self.config.theta_pass:
                    return AuditResult(
                        verdict="PASS",
                        reason="threshold_pass",
                        queries_used=t,
                        final_llr=float(state.llr),
                        final_p_compliant=float(state.p_compliant),
                        cumulative_leakage=float(cur_leakage),
                        target_id=str(target.target_id),
                        policy_name=policy_name,
                        steps=steps,
                        transcript=transcript,
                    )
                if state.llr <= self.config.theta_fail:
                    return AuditResult(
                        verdict="FAIL",
                        reason="threshold_fail",
                        queries_used=t,
                        final_llr=float(state.llr),
                        final_p_compliant=float(state.p_compliant),
                        cumulative_leakage=float(cur_leakage),
                        target_id=str(target.target_id),
                        policy_name=policy_name,
                        steps=steps,
                        transcript=transcript,
                    )

            # 2. Budget exhaustion checks
            if t >= max_b:
                # If budget exhausted, return INCONCLUSIVE
                # (Or if force_max_budget is active, decide based on sign of LLR at budget)
                if self.config.force_max_budget:
                    verdict = "PASS" if state.llr > 0 else "FAIL"
                    reason = "forced_budget_decision"
                else:
                    verdict = "INCONCLUSIVE"
                    reason = "query_budget_exhausted"

                return AuditResult(
                    verdict=verdict,
                    reason=reason,
                    queries_used=t,
                    final_llr=float(state.llr),
                    final_p_compliant=float(state.p_compliant),
                    cumulative_leakage=float(cur_leakage),
                    target_id=str(target.target_id),
                    policy_name=policy_name,
                    steps=steps,
                    transcript=transcript,
                )

            if np.isfinite(bp) and cur_leakage >= bp:
                return AuditResult(
                    verdict="INCONCLUSIVE",
                    reason="privacy_budget_exhausted",
                    queries_used=t,
                    final_llr=float(state.llr),
                    final_p_compliant=float(state.p_compliant),
                    cumulative_leakage=float(cur_leakage),
                    target_id=str(target.target_id),
                    policy_name=policy_name,
                    steps=steps,
                    transcript=transcript,
                )

            # 3. Policy query selection
            try:
                if hasattr(policy, "select_next"):
                    import inspect
                    sig = inspect.signature(policy.select_next)
                    kwargs = {}
                    if "state" in sig.parameters:
                        kwargs["state"] = state
                    if "candidate_ids" in sig.parameters:
                        kwargs["candidate_ids"] = candidate_ids
                    qid = policy.select_next(transcript, **kwargs)
                else:
                    raise RuntimeError("Policy does not implement select_next().")
            except NoFeasibleQueriesError:
                return AuditResult(
                    verdict="INCONCLUSIVE",
                    reason="privacy_budget_exhausted",
                    queries_used=t,
                    final_llr=float(state.llr),
                    final_p_compliant=float(state.p_compliant),
                    cumulative_leakage=float(cur_leakage),
                    target_id=str(target.target_id),
                    policy_name=policy_name,
                    steps=steps,
                    transcript=transcript,
                )
            except RuntimeError as e:
                # E.g. pool exhausted
                return AuditResult(
                    verdict="INCONCLUSIVE",
                    reason="pool_exhausted",
                    queries_used=t,
                    final_llr=float(state.llr),
                    final_p_compliant=float(state.p_compliant),
                    cumulative_leakage=float(cur_leakage),
                    target_id=str(target.target_id),
                    policy_name=policy_name,
                    steps=steps,
                    transcript=transcript,
                )

            # 4. Query target and record response
            response = target.query(qid)
            state = self.evidence_model.update(state, int(qid), response)

            # 5. Update cumulative leakage
            if privacy_estimator is not None:
                cur_leakage = privacy_estimator.cumulative_leakage(transcript.query_ids)
            elif hasattr(policy, "current_estimated_leakage"):
                cur_leakage = policy.current_estimated_leakage

            steps.append(
                AuditStepRecord(
                    step=t + 1,
                    query_id=int(qid),
                    llr=float(state.llr),
                    p_compliant=float(state.p_compliant),
                    cumulative_leakage=float(cur_leakage),
                )
            )
