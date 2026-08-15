"""Pure state machine for closed-loop CASK execution.

Neural network calls live outside this class. Keeping transitions pure makes
timeout, censoring, replanning, and safe-stop behavior independently testable.
"""

from __future__ import annotations

import dataclasses
import enum


class PolicyPhase(enum.Enum):
    PROPOSE = "propose"
    FEASIBILITY_CHECK = "feasibility_check"
    EXECUTE = "execute"
    VERIFY = "verify"
    SAFE_STOP = "safe_stop"


@dataclasses.dataclass(frozen=True)
class PolicyRuntime:
    phase: PolicyPhase = PolicyPhase.PROPOSE
    candidate_index: int | None = None
    replan_count: int = 0
    no_progress_count: int = 0
    active_skill: bool = False


@dataclasses.dataclass(frozen=True)
class VerifyResult:
    normal_termination: bool = False
    timeout: bool = False
    safety_triggered: bool = False
    made_progress: bool = True


class CaskPolicyStateMachine:
    def __init__(self, *, max_replans: int = 2, max_no_progress: int = 2):
        if max_replans < 0 or max_no_progress <= 0:
            raise ValueError("Invalid replanning or progress threshold.")
        self.max_replans = max_replans
        self.max_no_progress = max_no_progress

    def proposals_ready(self, runtime: PolicyRuntime, *, num_candidates: int) -> PolicyRuntime:
        self._require(runtime, PolicyPhase.PROPOSE)
        if num_candidates <= 0:
            return dataclasses.replace(runtime, phase=PolicyPhase.SAFE_STOP)
        return dataclasses.replace(
            runtime,
            phase=PolicyPhase.FEASIBILITY_CHECK,
            candidate_index=0,
        )

    def feasibility_result(
        self,
        runtime: PolicyRuntime,
        *,
        feasible: bool,
        num_candidates: int,
    ) -> PolicyRuntime:
        self._require(runtime, PolicyPhase.FEASIBILITY_CHECK)
        if feasible:
            return dataclasses.replace(
                runtime,
                phase=PolicyPhase.EXECUTE,
                active_skill=True,
            )
        next_candidate = (runtime.candidate_index or 0) + 1
        next_replan_count = runtime.replan_count + 1
        if next_candidate >= num_candidates or next_replan_count > self.max_replans:
            return dataclasses.replace(
                runtime,
                phase=PolicyPhase.SAFE_STOP,
                active_skill=False,
            )
        return dataclasses.replace(
            runtime,
            candidate_index=next_candidate,
            replan_count=next_replan_count,
        )

    def action_prefix_executed(self, runtime: PolicyRuntime) -> PolicyRuntime:
        self._require(runtime, PolicyPhase.EXECUTE)
        return dataclasses.replace(runtime, phase=PolicyPhase.VERIFY)

    def verification_result(
        self,
        runtime: PolicyRuntime,
        result: VerifyResult,
    ) -> PolicyRuntime:
        self._require(runtime, PolicyPhase.VERIFY)
        if result.safety_triggered:
            return dataclasses.replace(
                runtime,
                phase=PolicyPhase.SAFE_STOP,
                active_skill=False,
            )
        if result.normal_termination:
            return PolicyRuntime(phase=PolicyPhase.PROPOSE)

        no_progress_count = 0 if result.made_progress else runtime.no_progress_count + 1
        if no_progress_count >= self.max_no_progress:
            return dataclasses.replace(
                runtime,
                phase=PolicyPhase.SAFE_STOP,
                no_progress_count=no_progress_count,
                active_skill=False,
            )
        if result.timeout and not result.made_progress:
            return dataclasses.replace(
                runtime,
                phase=PolicyPhase.SAFE_STOP,
                no_progress_count=no_progress_count,
                active_skill=False,
            )
        return dataclasses.replace(
            runtime,
            phase=PolicyPhase.EXECUTE,
            no_progress_count=no_progress_count,
            active_skill=True,
        )

    @staticmethod
    def _require(runtime: PolicyRuntime, expected: PolicyPhase) -> None:
        if runtime.phase is not expected:
            raise ValueError(
                f"Transition requires phase {expected.value}, got {runtime.phase.value}."
            )
