from __future__ import annotations

from cask.policies.cask_policy import CaskPolicyStateMachine
from cask.policies.cask_policy import PolicyPhase
from cask.policies.cask_policy import PolicyRuntime
from cask.policies.cask_policy import VerifyResult


def test_normal_closed_loop_transition() -> None:
    machine = CaskPolicyStateMachine()
    runtime = machine.proposals_ready(PolicyRuntime(), num_candidates=3)
    runtime = machine.feasibility_result(runtime, feasible=True, num_candidates=3)
    assert runtime.phase is PolicyPhase.EXECUTE
    runtime = machine.action_prefix_executed(runtime)
    runtime = machine.verification_result(runtime, VerifyResult(made_progress=True))
    assert runtime.phase is PolicyPhase.EXECUTE
    runtime = machine.action_prefix_executed(runtime)
    runtime = machine.verification_result(
        runtime,
        VerifyResult(normal_termination=True),
    )
    assert runtime == PolicyRuntime(phase=PolicyPhase.PROPOSE)


def test_safety_trigger_always_stops() -> None:
    machine = CaskPolicyStateMachine()
    runtime = machine.proposals_ready(PolicyRuntime(), num_candidates=1)
    runtime = machine.feasibility_result(runtime, feasible=True, num_candidates=1)
    runtime = machine.action_prefix_executed(runtime)
    runtime = machine.verification_result(
        runtime,
        VerifyResult(safety_triggered=True),
    )
    assert runtime.phase is PolicyPhase.SAFE_STOP
