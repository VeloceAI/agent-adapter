import asyncio

import pytest
from pydantic import ValidationError

import agent_adapter as aa
from agent_adapter.errors import GuardrailBlocked
from agent_adapter.robotics import ActionSpec, Box, Command, KinematicSim, RobotGate, SimRobot
from agent_adapter.robotics.kinematic import straight_line
from agent_adapter.robotics.spec import judge

CELL = Box(lo=(-0.5, -0.5, 0.0), hi=(0.5, 0.5, 0.6))
SPEC = ActionSpec(workspace=CELL, max_speed=0.25, max_force=20, max_rate=50)
START = (0.0, 0.0, 0.1)


def move(speed: float, x: float = 0.1) -> Command:
    return Command(kind="move", target=(x, 0.0, 0.1), speed=speed)


def test_min_speed_defaults_to_a_thousandth_of_max_speed() -> None:
    assert SPEC.slowest_speed == pytest.approx(0.00025)
    assert (
        ActionSpec(workspace=CELL, max_speed=1, max_force=1, min_speed=0.01).slowest_speed == 0.01
    )


def test_min_speed_must_not_exceed_max_speed() -> None:
    with pytest.raises(ValidationError):
        ActionSpec(workspace=CELL, max_speed=0.25, max_force=1, min_speed=0.5)


def test_a_speed_below_the_minimum_is_blocked_not_approvable() -> None:
    verdict = judge(SPEC, move(1e-7))
    assert verdict.action == "block" and "below the minimum" in verdict.summary
    assert judge(SPEC, move(0.001)).action == "run"


def test_straight_line_refuses_too_many_waypoints() -> None:
    with pytest.raises(ValueError, match="waypoints"):
        straight_line(START, (0.4, 0.0, 0.1), 1e-7, 0.02)
    with pytest.raises(ValueError, match="waypoints"):
        straight_line(START, (0.4, 0.0, 0.1), 0.01, 0.02, max_steps=100)


def build() -> tuple[aa.Executor, SimRobot]:
    registry = aa.Registry()
    executor = aa.Executor(registry, approval=aa.CallbackApproval(lambda request: True))
    robot = SimRobot(START)
    RobotGate(SPEC, robot=robot, sim=KinematicSim()).add_tools(registry)
    return executor, robot


async def test_a_tiny_speed_is_blocked_at_once() -> None:
    executor, robot = build()
    args = {"x": 0.4, "y": 0.0, "z": 0.1, "speed": 1e-7}
    with pytest.raises(GuardrailBlocked, match="below the minimum"):
        await asyncio.wait_for(executor.run("move_to", args), 1.0)
    assert robot.log == []


async def test_a_move_over_the_duration_cap_is_blocked_before_simulation() -> None:
    class Watching(KinematicSim):
        calls = 0

        async def simulate(self, *args: object) -> list[object]:
            self.calls += 1
            return []

    registry = aa.Registry()
    executor = aa.Executor(registry)
    robot, sim = SimRobot(START), Watching()
    RobotGate(SPEC, robot=robot, sim=sim).add_tools(registry)
    args = {"x": 0.4, "y": 0.0, "z": 0.1, "speed": 0.001}  # 400 s, over the 60 s default
    with pytest.raises(GuardrailBlocked, match="max_duration"):
        await executor.run("move_to", args)
    assert robot.log == [] and sim.calls == 0


def test_max_duration_is_configurable() -> None:
    assert SPEC.max_duration == 60.0
    with pytest.raises(ValidationError):
        ActionSpec(workspace=CELL, max_speed=1, max_force=1, max_duration=0)
