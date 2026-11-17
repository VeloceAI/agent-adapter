"""ROS 2 robot port: velocity commands on a `cmd_vel` topic, position from a pose topic.

rclpy comes from a ROS 2 install, not from PyPI. Defaults fit turtlesim.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import Callable
from typing import Any

from agent_adapter.core.errors import MissingDependency, ToolError
from agent_adapter.robotics.kinematic import STOPPED
from agent_adapter.robotics.ports import _maybe
from agent_adapter.robotics.spec import Command, Point

__all__ = ["ROS2Robot", "odometry_pose", "turtlesim_pose"]

Pose2D = tuple[float, float, float]  # x, y, heading (rad)


def turtlesim_pose(msg: Any) -> Pose2D:
    """`turtlesim/msg/Pose` to (x, y, heading)."""
    return float(msg.x), float(msg.y), float(msg.theta)


def odometry_pose(msg: Any) -> Pose2D:
    """`nav_msgs/msg/Odometry` to (x, y, heading): the yaw of its orientation quaternion."""
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return float(p.x), float(p.y), yaw


def _ros() -> tuple[Any, Any, Any, Any]:
    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
    except ImportError as exc:
        raise MissingDependency(
            "ROS2Robot needs rclpy and geometry_msgs from a ROS 2 install (they are not on "
            "PyPI): source /opt/ros/<distro>/setup.bash first, or run the Docker example in "
            "examples/robotics/ros2"
        ) from exc
    return rclpy, Context, SingleThreadedExecutor, Twist


def _turtlesim_pose_type() -> Any:
    try:
        from turtlesim.msg import Pose
    except ImportError as exc:
        raise MissingDependency(
            "the default pose topic is turtlesim's: install ros-<distro>-turtlesim, or pass "
            "pose_type= and read_pose= for your robot"
        ) from exc
    return Pose


class ROS2Robot:
    """`RobotPort` for a ROS 2 mobile base driven by `geometry_msgs/Twist` velocity commands.

    Our `Point` maps to the robot's x and y in the pose topic's frame; z is fixed (`z=`) and
    a target with another z is refused. A move turns towards the target, then drives with
    `linear.x` never above the command's speed, until it is within `tolerance` metres; its
    final heading is not controlled. Missing pose updates for `pose_timeout` seconds or a
    move that overruns its deadline stops the robot with a `ToolError`.

    `stop` publishes a zero twist at once and latches: `execute` refuses until `reset()`.
    Without `node=` it opens its own rclpy context, node and spin thread (`close()` ends
    them); with `node=` you spin that node yourself. A grip runs `grip=` or is refused.
    """

    def __init__(
        self,
        *,
        node: Any = None,
        cmd_vel_topic: str = "/turtle1/cmd_vel",
        pose_topic: str = "/turtle1/pose",
        pose_type: Any = None,
        read_pose: Callable[[Any], Pose2D] = turtlesim_pose,
        z: float = 0.0,
        tolerance: float = 0.05,
        rate: float = 20.0,
        max_turn: float = 2.0,
        aligned: float = 0.1,
        pose_timeout: float = 1.0,
        grip: Callable[[float], Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        rclpy, context_type, executor_type, self._twist = _ros()
        if pose_type is None:
            pose_type = _turtlesim_pose_type()
        self.z, self.tolerance, self.period = z, tolerance, 1.0 / rate
        self.max_turn, self.aligned, self.pose_timeout = max_turn, aligned, pose_timeout
        self._read_pose, self._grip, self._clock = read_pose, grip, clock
        self._pose: tuple[Pose2D, float] | None = None  # one tuple: the spin thread swaps it
        self._got_pose = threading.Event()
        self._stop = asyncio.Event()
        self._rclpy, self._context, self._executor, self._thread = rclpy, None, None, None
        if node is None:
            self._context = context_type()
            rclpy.init(context=self._context)
            node = rclpy.create_node("agent_adapter_robot", context=self._context)
            self._executor = executor_type(context=self._context)
            self._executor.add_node(node)
            self._thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._thread.start()
        self.node = node
        self._publisher = node.create_publisher(self._twist, cmd_vel_topic, 10)
        self._subscription = node.create_subscription(pose_type, pose_topic, self._on_pose, 10)

    def _on_pose(self, msg: Any) -> None:
        self._pose = (self._read_pose(msg), self._clock())
        self._got_pose.set()

    async def wait_for_pose(self, timeout: float = 5.0) -> Point:
        """Wait for the first pose message; raises `ToolError` after `timeout` seconds."""
        if not await asyncio.to_thread(self._got_pose.wait, timeout):
            raise ToolError(f"no pose message within {timeout:g} s")
        return self.position()

    def position(self) -> Point:
        x, y, _ = self._pose2d()
        return (x, y, self.z)

    def _pose2d(self) -> Pose2D:
        if self._pose is None:
            raise ToolError("no pose message yet: the robot's position is unknown")
        return self._pose[0]

    async def execute(self, command: Command) -> None:
        if self._stop.is_set():
            raise ToolError(STOPPED)
        if command.kind == "grip":
            if self._grip is None:
                raise ToolError("this robot has no gripper")
            await _maybe(self._grip(command.force))
            return
        assert command.target is not None  # checked by Command
        tx, ty, tz = command.target
        if abs(tz - self.z) > self.tolerance:
            raise ToolError(f"this robot moves in x and y only: z must be {self.z:g}")
        try:
            await self._drive(tx, ty, command.speed)
        finally:
            self._send(0.0, 0.0)

    async def _drive(self, tx: float, ty: float, speed: float) -> None:
        x, y, _ = self._pose2d()
        turn_time = math.pi / self.max_turn
        deadline = self._clock() + 2 * math.hypot(tx - x, ty - y) / speed + turn_time + 2.0
        while not self._stop.is_set():
            assert self._pose is not None
            (x, y, heading), seen = self._pose
            now = self._clock()
            if now - seen > self.pose_timeout:
                raise ToolError(f"no pose for {self.pose_timeout:g} s: stopped")
            if now > deadline:
                raise ToolError(f"did not reach the target in time: stopped at {self._at()}")
            gap = math.hypot(tx - x, ty - y)
            if gap <= self.tolerance:
                return
            error = math.remainder(math.atan2(ty - y, tx - x) - heading, math.tau)
            turn = max(-self.max_turn, min(self.max_turn, 4.0 * error))
            forward = min(speed, 4.0 * gap) if abs(error) <= self.aligned else 0.0
            self._send(forward, turn)
            try:
                await asyncio.wait_for(self._stop.wait(), self.period)
            except TimeoutError:
                continue

    def _send(self, forward: float, turn: float) -> None:
        msg = self._twist()
        msg.linear.x, msg.angular.z = forward, turn
        self._publisher.publish(msg)

    def _at(self) -> str:
        return "(" + ", ".join(f"{v:.3f}" for v in self.position()) + ")"

    async def stop(self) -> None:
        self._stop.set()
        self._send(0.0, 0.0)

    def reset(self) -> None:
        self._stop.clear()

    def close(self) -> None:
        """Stop the spin thread and shut down the context this robot opened (if any)."""
        self._send(0.0, 0.0)
        if self._executor is None:
            return
        self._executor.shutdown()
        self.node.destroy_node()
        self._rclpy.try_shutdown(context=self._context)
        if self._thread is not None:
            self._thread.join(timeout=5)
