"""The two ports the gate drives: a shadow simulator and the robot itself."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from agent_adapter.robotics.spec import ActionSpec, Command, Point, Waypoint

__all__ = ["CallbackRobot", "RobotPort", "SimPort"]


@runtime_checkable
class SimPort(Protocol):
    """Runs a command in simulation first; returns the end effector's path."""

    async def simulate(
        self, start: Point, command: Command, spec: ActionSpec
    ) -> Sequence[Waypoint]:
        """The simulated path from `start`; an empty path for a command that does not move."""
        ...


@runtime_checkable
class RobotPort(Protocol):
    """A robot (or a stand-in) the gate sends commands to.

    `stop` halts motion now. It must be safe to call at any time, more than once, and while
    `execute` is still running: `execute` then returns early.
    """

    async def execute(self, command: Command) -> None: ...

    async def stop(self) -> None: ...

    def position(self) -> Point: ...


class CallbackRobot:
    """A `RobotPort` from your own functions: an rclpy publisher, a LeRobot robot, a driver.

    The callbacks may be sync or async. Nothing here imports ROS or LeRobot.
    """

    def __init__(
        self,
        *,
        execute: Callable[[Command], Any],
        stop: Callable[[], Any],
        position: Callable[[], Point],
    ) -> None:
        self._execute, self._stop, self._position = execute, stop, position

    async def execute(self, command: Command) -> None:
        await _maybe(self._execute(command))

    async def stop(self) -> None:
        await _maybe(self._stop())

    def position(self) -> Point:
        return self._position()


async def _maybe(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value
