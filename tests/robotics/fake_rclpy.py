"""A stand-in for rclpy, geometry_msgs and turtlesim: a turtle that follows cmd_vel."""

from __future__ import annotations

import math
import sys
import threading
import time
import types
from collections.abc import Callable
from typing import Any


class Vector3:
    def __init__(self) -> None:
        self.x = self.y = self.z = 0.0


class Twist:
    def __init__(self) -> None:
        self.linear, self.angular = Vector3(), Vector3()


class Pose:
    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self.x, self.y, self.theta = x, y, theta


class Publisher:
    def __init__(self, node: Node, topic: str) -> None:
        self.node, self.topic = node, topic

    def publish(self, msg: Twist) -> None:
        self.node.sent.append((self.topic, msg))
        if self.node.world is not None:
            self.node.world.command(msg)


class Node:
    def __init__(self, world: Turtle | None = None) -> None:
        self.world = world
        self.sent: list[tuple[str, Twist]] = []
        self.callbacks: dict[str, Callable[[Any], None]] = {}
        self.destroyed = False

    def create_publisher(self, msg_type: Any, topic: str, qos: int) -> Publisher:
        return Publisher(self, topic)

    def create_subscription(
        self, msg_type: Any, topic: str, callback: Callable[[Any], None], qos: int
    ) -> object:
        self.callbacks[topic] = callback
        return object()

    def destroy_node(self) -> None:
        self.destroyed = True


class Turtle:
    """Integrates the last twist in real time, like turtlesim's update loop."""

    def __init__(self, x: float = 5.5, y: float = 5.5, theta: float = 0.0) -> None:
        self.x, self.y, self.theta = x, y, theta
        self.forward = self.turn = 0.0
        self.top_speed = 0.0
        self._last = time.perf_counter()
        self._lock = threading.Lock()

    def command(self, msg: Twist) -> None:
        with self._lock:
            self.forward, self.turn = msg.linear.x, msg.angular.z
            self.top_speed = max(self.top_speed, abs(self.forward))

    def step(self) -> Pose:
        with self._lock:
            now = time.perf_counter()
            dt, self._last = now - self._last, now
            self.theta = math.remainder(self.theta + self.turn * dt, math.tau)
            self.x += math.cos(self.theta) * self.forward * dt
            self.y += math.sin(self.theta) * self.forward * dt
            return Pose(self.x, self.y, self.theta)


class World:
    """Holds what the fake modules created, for the test to inspect."""

    def __init__(self) -> None:
        self.turtle = Turtle()
        self.nodes: list[Node] = []
        self.executors: list[Executor] = []
        self.publish_pose = True


class Executor:
    def __init__(self, world: World, context: object) -> None:
        self.world, self.context = world, context
        self.nodes: list[Node] = []
        self._done = threading.Event()
        world.executors.append(self)

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def spin(self) -> None:
        while not self._done.wait(0.002):
            pose = self.world.turtle.step()
            if self.world.publish_pose:
                for node in self.nodes:
                    for callback in node.callbacks.values():
                        callback(pose)

    def shutdown(self) -> None:
        self._done.set()


def install(monkeypatch: Any, *, turtlesim: bool = True) -> World:
    world = World()
    rclpy = types.ModuleType("rclpy")
    context = types.ModuleType("rclpy.context")
    executors = types.ModuleType("rclpy.executors")
    context.Context = type("Context", (), {})  # type: ignore[attr-defined]
    executors.SingleThreadedExecutor = lambda context: Executor(world, context)  # type: ignore[attr-defined]

    def create_node(name: str, *, context: object) -> Node:
        node = Node(world.turtle)
        world.nodes.append(node)
        return node

    rclpy.init = lambda context: None  # type: ignore[attr-defined]
    rclpy.try_shutdown = lambda context: None  # type: ignore[attr-defined]
    rclpy.create_node = create_node  # type: ignore[attr-defined]
    geometry = types.ModuleType("geometry_msgs.msg")
    geometry.Twist = Twist  # type: ignore[attr-defined]
    modules = {
        "rclpy": rclpy,
        "rclpy.context": context,
        "rclpy.executors": executors,
        "geometry_msgs": types.ModuleType("geometry_msgs"),
        "geometry_msgs.msg": geometry,
    }
    if turtlesim:
        msg = types.ModuleType("turtlesim.msg")
        msg.Pose = Pose  # type: ignore[attr-defined]
        modules |= {"turtlesim": types.ModuleType("turtlesim"), "turtlesim.msg": msg}
    else:
        modules |= {"turtlesim": None, "turtlesim.msg": None}  # type: ignore[dict-item]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return world
