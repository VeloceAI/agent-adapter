import asyncio
from typing import Any

import pytest

import agent_adapter as aa
from agent_adapter.errors import GuardrailBlocked, ToolError
from agent_adapter.exec import changes_things
from agent_adapter.robotics import (
    ActionSpec,
    Box,
    CallbackRobot,
    Command,
    KinematicSim,
    RobotGate,
    SimRobot,
)
from agent_adapter.robotics.spec import Point

SPEC = ActionSpec(
    workspace=Box(lo=(-1, -1, -1), hi=(1, 1, 1)), max_speed=2.0, max_force=10, max_rate=50
)
START = (0.0, 0.0, 0.0)


class SlowSim(KinematicSim):
    """A shadow simulation that yields to the loop, like MujocoSim's worker thread."""

    def __init__(self, during: Any = None) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.during = during

    async def simulate(self, start: Point, command: Command, spec: ActionSpec) -> Any:
        self.started.set()
        await asyncio.sleep(0.1)
        if self.during is not None:
            self.during()
        return await super().simulate(start, command, spec)


def build(sim: Any = None, **gate_kw: Any) -> tuple[RobotGate, aa.Executor, SimRobot]:
    registry = aa.Registry()
    executor = aa.Executor(registry, approval=aa.CallbackApproval(lambda request: True))
    robot = SimRobot(START)
    gate = RobotGate(SPEC, robot=robot, sim=sim, executor=executor, **gate_kw)
    gate.add_tools(registry)
    return gate, executor, robot


def move(x: float, speed: float = 1.0) -> dict[str, float]:
    return {"x": x, "y": 0.0, "z": 0.0, "speed": speed}


async def test_estop_during_the_shadow_simulation_blocks_the_move() -> None:
    sim = SlowSim()
    gate, executor, robot = build(sim=sim)
    task = asyncio.create_task(executor.run("move_to", move(0.5)))
    await sim.started.wait()
    await gate.estop("operator pressed stop")
    with pytest.raises(GuardrailBlocked, match="e-stop"):
        await task
    assert robot.log == [] and robot.at == START


async def test_estop_and_reset_during_the_simulation_still_blocks() -> None:
    sim = SlowSim()
    gate, executor, robot = build(sim=sim)
    task = asyncio.create_task(executor.run("move_to", move(0.5)))
    await sim.started.wait()
    await gate.estop("tap")
    gate.reset()
    with pytest.raises(GuardrailBlocked, match="e-stop"):
        await task
    assert robot.log == []


async def test_heartbeat_going_stale_during_the_simulation_blocks_the_move() -> None:
    now = [0.0]
    sim = SlowSim(during=lambda: now.__setitem__(0, 5.0))
    gate, executor, robot = build(sim=sim, clock=lambda: now[0])
    async with gate.deadman(1.0):
        with pytest.raises(GuardrailBlocked, match="heartbeat"):
            await executor.run("move_to", move(0.5))
    assert robot.log == []


async def test_sim_robot_refuses_to_move_while_stopped_until_reset() -> None:
    robot = SimRobot(START)
    await robot.stop()
    with pytest.raises(ToolError, match="stopped"):
        await robot.execute(Command(kind="move", target=(0.1, 0, 0), speed=1.0))
    with pytest.raises(ToolError, match="stopped"):
        await robot.execute(Command(kind="grip", force=1))
    assert robot.log == [] and robot.at == START
    robot.reset()
    await robot.execute(Command(kind="move", target=(0.1, 0, 0), speed=10.0))
    assert robot.at == (0.1, 0, 0)


async def test_callback_robot_refuses_while_stopped_until_reset() -> None:
    seen: list[Any] = []
    resets: list[int] = []
    robot = CallbackRobot(
        execute=seen.append,
        stop=lambda: None,
        position=lambda: START,
        reset=lambda: resets.append(1),
    )
    await robot.stop()
    with pytest.raises(ToolError, match="stopped"):
        await robot.execute(Command(kind="grip", force=1))
    robot.reset()
    await robot.execute(Command(kind="grip", force=1))
    assert len(seen) == 1 and resets == [1]


async def test_gate_reset_rearms_the_robot() -> None:
    gate, executor, robot = build()
    await gate.estop("tap")
    assert await executor.run("reset_estop") == "e-stop released"
    assert "done" in await executor.run("move_to", move(0.1, speed=2.0))
    assert robot.at == (0.1, 0.0, 0.0)


def test_motion_is_not_shielded_from_cancellation() -> None:
    registry = aa.Registry()
    RobotGate(SPEC, robot=SimRobot()).add_tools(registry)
    assert not changes_things(registry.get("move_to"))
    assert not changes_things(registry.get("grip"))


async def test_cancelling_a_running_move_stops_the_robot_and_latches() -> None:
    gate, executor, robot = build()
    task = asyncio.create_task(executor.run("move_to", move(0.9, speed=0.2)))
    await asyncio.sleep(0.15)
    assert robot.moving
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 0.5)
    assert not robot.moving and robot.at[0] < 0.9
    assert gate.latched is not None and "cancel" in gate.latched


async def test_estop_tool_in_the_same_run_does_not_wait_for_the_move() -> None:
    gate, executor, robot = build()
    async with executor.run_scope() as ctx:
        task = asyncio.create_task(executor.run("move_to", move(0.9, speed=0.2), ctx))
        await asyncio.sleep(0.15)
        assert robot.moving
        text = await asyncio.wait_for(executor.run("estop", {"reason": "voice"}, ctx), 0.5)
        assert "stopped" in text
        with pytest.raises(ToolError, match="e-stop"):
            await asyncio.wait_for(task, 0.5)
    assert gate.latched == "voice" and not robot.moving and robot.at[0] < 0.9


async def test_moves_in_one_run_still_run_one_at_a_time() -> None:
    class Counting(SimRobot):
        running = peak = 0

        async def execute(self, command: Command) -> None:
            self.running += 1
            self.peak = max(self.peak, self.running)
            try:
                await super().execute(command)
            finally:
                self.running -= 1

    registry = aa.Registry()
    executor = aa.Executor(registry)
    robot = Counting(START)
    RobotGate(SPEC, robot=robot).add_tools(registry)
    calls = [aa.ToolCall(name="move_to", arguments=move(x, speed=2.0)) for x in (0.1, 0.2, 0.3)]
    async with executor.run_scope() as ctx:
        results = await executor.run_many(calls, ctx)
    assert all(r.ok for r in results) and len(robot.log) == 3 and robot.peak == 1
