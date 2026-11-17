"""Drive turtlesim through the robot gate: a move, an e-stop mid-move, a refusal, a reset.

Simulation only. Run inside the Docker image next to this file (see its Dockerfile).
"""

import asyncio
import math
import sys

import agent_adapter as aa
from agent_adapter import ToolError
from agent_adapter.errors import GuardrailBlocked
from agent_adapter.robotics import ActionSpec, Box, RobotGate
from agent_adapter.robotics.ros2 import ROS2Robot

# turtlesim's window is 11 x 11 units; our Point is (x, y, z) with z fixed at 0
spec = ActionSpec(
    workspace=Box(lo=(0.5, 0.5, 0.0), hi=(10.5, 10.5, 0.0)),
    max_speed=1.5,
    max_force=0,
    max_duration=30,
)


def reviewer(request: aa.ApprovalRequest) -> bool:  # a person: releases the e-stop, nothing else
    print(f"  asked: {request.tool} -> {'yes' if request.tool == 'reset_estop' else 'no'}")
    return request.tool == "reset_estop"


def at(robot: ROS2Robot) -> str:
    x, y, _ = robot.position()
    return f"({x:.2f}, {y:.2f})"


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}: {what}")
    if not ok:
        sys.exit(1)


async def main() -> None:
    robot = ROS2Robot()  # /turtle1/cmd_vel and /turtle1/pose
    registry = aa.Registry()
    executor = aa.Executor(registry, approval=aa.CallbackApproval(reviewer))
    gate = RobotGate(spec, robot=robot, executor=executor)
    gate.add_tools(registry)
    try:
        await robot.wait_for_pose(15)
        print(f"turtle at {at(robot)}")

        print(await executor.run("move_to", {"x": 8.0, "y": 5.5, "z": 0.0, "speed": 1.0}))
        x, y, _ = robot.position()
        check(math.hypot(x - 8.0, y - 5.5) <= 0.1, f"move reached (8.00, 5.50): {at(robot)}")

        try:
            await executor.run("move_to", {"x": 3.0, "y": 5.5, "z": 0.0, "speed": 3.0})
            check(False, "a move over max_speed needs a person")
        except aa.ApprovalDenied:
            check(True, "a move over max_speed went to the reviewer, who said no")

        move = asyncio.create_task(
            executor.run("move_to", {"x": 2.0, "y": 2.0, "z": 0.0, "speed": 1.0})
        )
        await asyncio.sleep(3.0)
        await gate.estop("operator pressed stop")
        at_stop = robot.position()
        try:
            await move
            check(False, "estop interrupts the move")
        except ToolError as exc:
            print(f"  {exc}")
        await asyncio.sleep(0.2)  # the zero twist lands on turtlesim's next 16 ms tick
        halted = robot.position()
        await asyncio.sleep(1.0)  # turtlesim keeps publishing its pose
        drift = math.dist(halted, robot.position())
        coast = math.dist(at_stop, halted)
        far = math.dist(halted, (2.0, 2.0, 0.0))
        check(far > 1.0 and coast < 0.1, f"stop halted it at {at(robot)}, {coast:.3f} after stop")
        check(drift < 0.001, f"it stays put: moved {drift:.4f} in the next second")

        try:
            await executor.run("move_to", {"x": 5.0, "y": 5.0, "z": 0.0, "speed": 1.0})
            check(False, "a move after the e-stop is refused")
        except GuardrailBlocked as exc:
            check(True, f"a move after the e-stop is refused: {exc.reason}")

        print(await executor.run("reset_estop", {}))
        print(await executor.run("move_to", {"x": 3.0, "y": 8.0, "z": 0.0, "speed": 1.0}))
        x, y, _ = robot.position()
        check(math.hypot(x - 3.0, y - 8.0) <= 0.1, f"after reset it moves again: {at(robot)}")
    finally:
        robot.close()


asyncio.run(main())
