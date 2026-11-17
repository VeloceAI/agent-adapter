import asyncio
import math
import sys
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

import agent_adapter as aa
from agent_adapter.errors import GuardrailBlocked, MissingDependency, ToolError
from agent_adapter.robotics import ActionSpec, Box, Command, RobotGate, RobotPort
from agent_adapter.robotics.ros2 import ROS2Robot, odometry_pose, turtlesim_pose
from tests.robotics import fake_rclpy

SPEC = ActionSpec(
    workspace=Box(lo=(0.0, 0.0, 0.0), hi=(11.0, 11.0, 0.0)),
    max_speed=2.0,
    max_force=0,
    max_rate=50,
)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> fake_rclpy.World:
    return fake_rclpy.install(monkeypatch)


@pytest.fixture
def robot(world: fake_rclpy.World) -> Iterator[ROS2Robot]:
    robot = ROS2Robot(rate=200)
    yield robot
    robot.close()


def gated(robot: ROS2Robot) -> tuple[aa.Executor, RobotGate]:
    registry = aa.Registry()
    executor = aa.Executor(registry)
    gate = RobotGate(SPEC, robot=robot, executor=executor)
    gate.add_tools(registry)
    return executor, gate


def move(x: float, y: float, speed: float = 2.0) -> dict[str, float]:
    return {"x": x, "y": y, "z": 0.0, "speed": speed}


def test_missing_rclpy_says_where_it_comes_from(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rclpy", None)
    with pytest.raises(MissingDependency, match=r"ROS 2 install .*not on PyPI"):
        ROS2Robot()


def test_missing_turtlesim_needs_a_pose_type(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_rclpy.install(monkeypatch, turtlesim=False)
    with pytest.raises(MissingDependency, match="pose_type="):
        ROS2Robot()
    ROS2Robot(pose_type=object, read_pose=odometry_pose).close()


async def test_a_gated_move_reaches_the_target(robot: ROS2Robot, world: fake_rclpy.World) -> None:
    assert isinstance(robot, RobotPort)
    assert await robot.wait_for_pose() == pytest.approx((5.5, 5.5, 0.0))
    executor, _ = gated(robot)
    done = await executor.run("move_to", move(6.0, 6.0, speed=1.5))
    x, y, z = robot.position()
    assert math.hypot(x - 6.0, y - 6.0) <= robot.tolerance + 0.02
    assert z == 0.0 and done.startswith("done: move")
    assert world.turtle.top_speed <= 1.5
    topic, last = world.nodes[0].sent[-1]
    assert topic == "/turtle1/cmd_vel" and last.linear.x == last.angular.z == 0.0


async def test_estop_halts_a_move_and_reset_allows_it_again(
    robot: ROS2Robot, world: fake_rclpy.World
) -> None:
    await robot.wait_for_pose()
    executor, gate = gated(robot)
    running = asyncio.create_task(executor.run("move_to", move(9.0, 5.5, speed=1.0)))
    await asyncio.sleep(0.3)
    await gate.estop("operator")
    with pytest.raises(ToolError, match=r"e-stop \(operator\)"):
        await running
    halted = robot.position()
    assert 5.6 < halted[0] < 8.0
    assert world.turtle.forward == world.turtle.turn == 0.0
    await asyncio.sleep(0.05)
    assert robot.position()[0] == pytest.approx(halted[0], abs=1e-6)

    with pytest.raises(GuardrailBlocked, match="e-stop is latched"):
        await executor.run("move_to", move(5.5, 5.5))
    with pytest.raises(ToolError, match="reset it"):
        await robot.execute(Command(kind="move", target=(5.5, 5.5, 0.0), speed=1.0))

    gate.reset()
    await executor.run("move_to", move(halted[0] + 0.3, 5.5))
    assert robot.position()[0] == pytest.approx(halted[0] + 0.3, abs=0.07)


async def test_z_is_fixed_and_a_grip_needs_a_gripper(robot: ROS2Robot) -> None:
    await robot.wait_for_pose()
    with pytest.raises(ToolError, match="x and y only"):
        await robot.execute(Command(kind="move", target=(6.0, 6.0, 0.5), speed=1.0))
    with pytest.raises(ToolError, match="no gripper"):
        await robot.execute(Command(kind="grip", force=1))


async def test_grip_runs_the_callback(world: fake_rclpy.World) -> None:
    forces: list[float] = []

    async def grip(force: float) -> None:
        forces.append(force)

    robot = ROS2Robot(grip=grip)
    try:
        await robot.execute(Command(kind="grip", force=3))
    finally:
        robot.close()
    assert forces == [3]


async def test_lost_pose_stops_the_move(world: fake_rclpy.World) -> None:
    node = fake_rclpy.Node()
    robot = ROS2Robot(node=node, cmd_vel_topic="/cmd_vel", pose_topic="/pose", pose_timeout=0.1)
    with pytest.raises(ToolError, match="position is unknown"):
        robot.position()
    with pytest.raises(ToolError, match="no pose message within"):
        await robot.wait_for_pose(timeout=0.01)
    node.callbacks["/pose"](fake_rclpy.Pose(1.0, 1.0, 0.0))
    with pytest.raises(ToolError, match=r"no pose for 0.1 s"):
        await robot.execute(Command(kind="move", target=(3.0, 1.0, 0.0), speed=1.0))
    topic, last = node.sent[-1]
    assert topic == "/cmd_vel" and last.linear.x == 0.0
    robot.close()
    assert not node.destroyed  # a node you pass in stays yours


def test_close_ends_the_spin_thread(world: fake_rclpy.World) -> None:
    robot = ROS2Robot()
    robot.close()
    assert world.nodes[0].destroyed
    assert robot._thread is not None and not robot._thread.is_alive()


def test_pose_readers() -> None:
    assert turtlesim_pose(fake_rclpy.Pose(1, 2, 0.5)) == (1.0, 2.0, 0.5)
    half_turn = SimpleNamespace(x=0.0, y=0.0, z=math.sin(math.pi / 4), w=math.cos(math.pi / 4))
    odom = SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(position=SimpleNamespace(x=3, y=4, z=0), orientation=half_turn)
        )
    )
    assert odometry_pose(odom) == pytest.approx((3.0, 4.0, math.pi / 2))
