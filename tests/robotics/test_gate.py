import asyncio
import math
from typing import Any

import pytest

import agent_adapter as aa
from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.errors import GuardrailBlocked, RunLimitExceeded, ToolError, ToolUsageError
from agent_adapter.robotics import (
    ActionSpec,
    Box,
    CallbackRobot,
    Command,
    KinematicSim,
    RobotGate,
    SimRobot,
)
from agent_adapter.tracing import EventLog

SPEC = ActionSpec(
    workspace=Box(lo=(-0.5, -0.5, 0.0), hi=(0.5, 0.5, 0.6)),
    max_speed=0.25,
    max_force=20,
    max_rate=50,
    keep_out=(Box(lo=(0.2, -0.1, 0.0), hi=(0.3, 0.1, 0.3), name="operator"),),
    hard_max_speed=1.0,
)
START = (0.0, 0.0, 0.1)


class Reviewer:
    def __init__(self, answer: bool) -> None:
        self.answer, self.asked = answer, []

    def __call__(self, request: aa.ApprovalRequest) -> bool:
        self.asked.append(request)
        return self.answer


def build(
    answer: bool | None = None, *, spec: ActionSpec = SPEC, sim: Any = None, **gate_kw: Any
) -> tuple[RobotGate, aa.Executor, SimRobot, Reviewer | None, EventLog]:
    registry = aa.Registry()
    reviewer = None if answer is None else Reviewer(answer)
    approval = None if reviewer is None else aa.CallbackApproval(reviewer)
    events = EventLog()
    executor = aa.Executor(registry, approval=approval, sink=events)
    robot = SimRobot(START)
    gate = RobotGate(spec, robot=robot, sim=sim, executor=executor, **gate_kw)
    gate.add_tools(registry)
    return gate, executor, robot, reviewer, events


def at(x: float, y: float, z: float, speed: float = 0.2) -> dict[str, float]:
    return {"x": x, "y": y, "z": z, "speed": speed}


async def test_inside_envelope_runs_without_asking() -> None:
    _, executor, robot, reviewer, events = build(answer=False)
    assert "done" in await executor.run("move_to", at(0.1, 0.1, 0.1))
    assert robot.at == (0.1, 0.1, 0.1)
    assert reviewer is not None and reviewer.asked == []
    assert [e.data["action"] for e in events.of_type("custom.robot_check")] == ["run"]
    assert events.of_type("custom.robot_done")
    assert events.of_type("tool.finished")


async def test_outside_envelope_asks_the_executor_port() -> None:
    _, executor, robot, reviewer, _ = build(answer=False)
    with pytest.raises(aa.ApprovalDenied):
        await executor.run("move_to", at(0.1, 0.1, 0.1, speed=0.5))
    assert robot.log == [] and reviewer is not None and len(reviewer.asked) == 1

    _, executor, robot, reviewer, _ = build(answer=True)
    await executor.run("move_to", at(-0.45, 0.0, 0.5, speed=0.6))
    assert robot.at == (-0.45, 0.0, 0.5)


async def test_outside_envelope_without_a_port_is_denied() -> None:
    _, executor, robot, _, _ = build(answer=None)
    with pytest.raises(aa.ApprovalDenied):
        await executor.run("grip", {"force": 50})
    assert robot.log == []


async def test_hard_limit_blocks_and_never_asks() -> None:
    _, executor, robot, reviewer, _ = build(answer=True)
    with pytest.raises(GuardrailBlocked, match="hard limit"):
        await executor.run("move_to", at(0.1, 0.1, 0.1, speed=3.0))
    assert robot.log == [] and reviewer is not None and reviewer.asked == []


async def test_simulated_path_through_keep_out_is_blocked() -> None:
    _, executor, robot, reviewer, _ = build(answer=True)
    with pytest.raises(GuardrailBlocked, match="operator"):
        await executor.run("move_to", at(0.45, 0.0, 0.1))
    assert robot.log == [] and reviewer is not None and reviewer.asked == []


async def test_invalid_numbers_are_blocked() -> None:
    _, executor, robot, _, _ = build(answer=True)
    with pytest.raises(GuardrailBlocked, match="invalid move"):
        await executor.run("move_to", at(math.nan, 0, 0.1))
    assert robot.log == []


async def test_rate_limit_blocks() -> None:
    now = [0.0]
    spec = SPEC.model_copy(update={"max_rate": 2})
    _, executor, robot, _, _ = build(answer=True, spec=spec, clock=lambda: now[0])
    await executor.run("grip", {"force": 1})
    await executor.run("grip", {"force": 2})
    with pytest.raises(GuardrailBlocked, match="per second"):
        await executor.run("grip", {"force": 3})
    now[0] = 1.5
    await executor.run("grip", {"force": 4})
    assert [c.force for c in robot.log] == [1, 2, 4]


async def test_a_direct_call_is_refused() -> None:
    _, executor, robot, _, _ = build(answer=True)
    spec = executor.registry.get("move_to")
    with pytest.raises(ToolUsageError, match="only through the executor"):
        await spec.invoke(at(0.1, 0, 0.1))
    with pytest.raises(ToolUsageError):
        await executor.registry.get("reset_estop").invoke({})
    assert robot.log == []


async def test_shadow_simulation_failure_blocks() -> None:
    class Broken:
        async def simulate(self, *args: Any) -> list[Any]:
            raise RuntimeError("boom")

    _, executor, robot, _, _ = build(answer=True, sim=Broken())
    with pytest.raises(GuardrailBlocked, match="shadow simulation failed"):
        await executor.run("move_to", at(0.1, 0, 0.1))
    assert robot.log == []


async def test_deferred_approval_resumes_once() -> None:
    registry = aa.Registry()
    executor = aa.Executor(registry, approval=aa.DeferredApproval(MemoryApprovalStore()))
    robot = SimRobot(START)
    RobotGate(SPEC, robot=robot).add_tools(registry)
    with pytest.raises(aa.ApprovalPending) as pending:
        await executor.run("grip", {"force": 30})
    assert robot.log == []
    result = await executor.resume(pending.value.approval_id, True, decided_by="ana")
    assert result.ok and robot.force == 30


async def test_estop_halts_mid_motion_and_latches() -> None:
    gate, executor, robot, reviewer, events = build(answer=True)
    move = asyncio.create_task(executor.run("move_to", at(0.0, 0.45, 0.1, speed=0.2)))
    await asyncio.sleep(0.2)
    assert robot.moving
    await gate.estop("operator pressed stop")
    with pytest.raises(ToolError, match="e-stop"):
        await move
    assert not robot.moving and robot.at[1] < 0.45
    assert gate.latched == "operator pressed stop"
    assert events.of_type("custom.robot_estop")[0].data["reason"] == "operator pressed stop"

    with pytest.raises(GuardrailBlocked, match="latched"):
        await executor.run("move_to", at(0, 0, 0.1))
    assert await executor.run("reset_estop") == "e-stop released"
    assert reviewer is not None and reviewer.asked[-1].tool == "reset_estop"
    assert gate.latched is None
    assert events.of_type("custom.robot_reset")
    await executor.run("move_to", at(0, 0, 0.1))


async def test_estop_tool_from_another_run_stops_a_running_move() -> None:
    gate, executor, robot, _, _ = build(answer=True)
    move = asyncio.create_task(executor.run("move_to", at(0.0, 0.45, 0.1)))
    await asyncio.sleep(0.1)
    text = await asyncio.wait_for(executor.run("estop", {"reason": "supervisor"}), 0.5)
    assert "stopped" in text
    with pytest.raises(ToolError):
        await move
    assert gate.latched == "supervisor" and not robot.moving


async def test_estop_works_when_the_executor_refuses_calls() -> None:
    registry = aa.Registry()
    executor = aa.Executor(registry, limits=aa.RunLimits(max_calls=1))
    robot = SimRobot(START)
    gate = RobotGate(SPEC, robot=robot, executor=executor)
    gate.add_tools(registry)
    async with executor.run_scope() as ctx:
        await executor.run("grip", {"force": 5}, ctx)
        with pytest.raises(RunLimitExceeded):
            await executor.run("estop", {}, ctx)
    await gate.estop()
    assert gate.latched == "estop" and robot.force == 0


async def test_deadman_stops_motion_when_heartbeats_stop() -> None:
    gate, executor, robot, _, _ = build(answer=True)
    async with gate.deadman(0.3):
        for _ in range(5):
            await asyncio.sleep(0.03)
            gate.heartbeat()
        assert gate.latched is None
        move = asyncio.create_task(executor.run("move_to", at(0.0, 0.45, 0.1)))
        with pytest.raises(ToolError, match="heartbeat lost"):
            await asyncio.wait_for(move, 2.0)
    assert not robot.moving and robot.at[1] < 0.45
    gate.heartbeat()  # no deadman armed: a no-op


async def test_stale_heartbeat_blocks_before_the_watchdog_fires() -> None:
    now = [0.0]
    gate, _, _, _, _ = build(answer=True, clock=lambda: now[0])
    async with gate.deadman(0.5):
        now[0] = 1.0
        verdict = await gate.check(Command(kind="grip", force=1))
    assert verdict.summary == "no operator heartbeat"


async def test_reset_rearms_a_tripped_deadman() -> None:
    gate, _, _, _, _ = build(answer=True)
    async with gate.deadman(0.05) as watchdog:
        await asyncio.sleep(0.15)
        assert watchdog.tripped and gate.latched == "heartbeat lost"
        gate.reset()
        assert watchdog.alive() and gate.latched is None


async def test_callback_robot_wraps_sync_and_async_functions() -> None:
    seen: list[Any] = []

    async def stop() -> None:
        seen.append("stop")

    robot = CallbackRobot(execute=seen.append, stop=stop, position=lambda: (1.0, 2.0, 3.0))
    await robot.execute(Command(kind="grip", force=1))
    await robot.stop()
    assert robot.position() == (1.0, 2.0, 3.0) and seen[1] == "stop"


async def test_kinematic_sim_paths() -> None:
    sim = KinematicSim(dt=0.1)
    assert await sim.simulate(START, Command(kind="grip", force=1), SPEC) == []
    still = await sim.simulate(START, Command(kind="move", target=START, speed=1), SPEC)
    assert [w.at for w in still] == [START]
    path = await sim.simulate(START, Command(kind="move", target=(0, 0.2, 0.1), speed=0.5), SPEC)
    assert path[-1].at == (0, 0.2, 0.1) and path[-1].t == pytest.approx(0.4)


async def test_cancelling_the_caller_lets_a_started_move_finish() -> None:
    _, executor, robot, _, _ = build(answer=True)
    move = asyncio.create_task(executor.run("move_to", at(0.0, 0.1, 0.1, speed=0.25)))
    await asyncio.sleep(0.1)
    move.cancel()
    with pytest.raises(asyncio.CancelledError):
        await move
    assert robot.at == (0.0, 0.1, 0.1) and not robot.moving
