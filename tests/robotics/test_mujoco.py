import sys

import pytest

import agent_adapter as aa
from agent_adapter.errors import GuardrailBlocked, MissingDependency
from agent_adapter.robotics import ActionSpec, Box, Command, RobotGate, SimRobot

SPEC = ActionSpec(
    workspace=Box(lo=(-0.5, -0.5, 0.0), hi=(0.5, 0.5, 0.6)),
    max_speed=0.25,
    max_force=20,
    keep_out=(Box(lo=(0.2, -0.1, 0.0), hi=(0.3, 0.1, 0.3), name="operator"),),
)
START = (0.0, 0.0, 0.1)


def test_missing_mujoco_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_adapter.robotics import mujoco_sim

    monkeypatch.setitem(sys.modules, "mujoco", None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[mujoco\]"):
        mujoco_sim.MujocoSim()


async def test_mujoco_path_follows_the_command() -> None:
    pytest.importorskip("mujoco")
    from agent_adapter.robotics.mujoco_sim import MujocoSim

    sim = MujocoSim()
    assert await sim.simulate(START, Command(kind="grip", force=1), SPEC) == []
    target = (0.0, 0.3, 0.2)
    path = await sim.simulate(START, Command(kind="move", target=target, speed=0.2), SPEC)
    end = path[-1].at
    assert all(abs(a - b) < 0.01 for a, b in zip(end, target, strict=True))
    assert path[-1].t > 1.0  # 0.22 m at 0.2 m/s plus settling


async def test_gate_blocks_a_keep_out_path_in_mujoco() -> None:
    pytest.importorskip("mujoco")
    from agent_adapter.robotics.mujoco_sim import MujocoSim

    registry = aa.Registry()
    executor = aa.Executor(registry, approval=aa.AutoApprove())
    robot = SimRobot(START)
    RobotGate(SPEC, robot=robot, sim=MujocoSim()).add_tools(registry)
    with pytest.raises(GuardrailBlocked, match="operator"):
        await executor.run("move_to", {"x": 0.45, "y": 0.0, "z": 0.1, "speed": 0.2})
    assert robot.log == []
    await executor.run("move_to", {"x": 0.0, "y": 0.1, "z": 0.1, "speed": 0.2})
    assert robot.at == (0.0, 0.1, 0.1)
