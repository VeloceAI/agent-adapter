"""A built-in kinematic simulator and a simulated robot: no physics engine, no hardware."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable

from agent_adapter.robotics.spec import ActionSpec, Command, Point, Waypoint

__all__ = ["KinematicSim", "SimRobot", "straight_line"]


def straight_line(start: Point, target: Point, speed: float, dt: float) -> list[Waypoint]:
    """Constant-speed samples every `dt` seconds from `start` to `target` (the last is exact)."""
    distance = math.dist(start, target)
    if distance == 0:
        return [Waypoint(t=0.0, at=target)]
    duration = distance / speed
    steps = max(1, math.ceil(duration / dt))
    path = []
    for i in range(1, steps + 1):
        f = i / steps
        at = tuple(a + (b - a) * f for a, b in zip(start, target, strict=True))
        path.append(Waypoint(t=duration * f, at=at))
    return path


class KinematicSim:
    """`SimPort`: the end effector moves in a straight line at the commanded speed."""

    def __init__(self, *, dt: float = 0.02) -> None:
        self.dt = dt

    async def simulate(self, start: Point, command: Command, spec: ActionSpec) -> list[Waypoint]:
        if command.kind != "move" or command.target is None:
            return []
        return straight_line(start, command.target, command.speed, self.dt)


class SimRobot:
    """`RobotPort` stand-in that moves a point in real time, one control tick at a time.

    `stop` wakes the control loop at once. `log` keeps every command that reached the
    robot; `halted_at` is the clock time the last motion ended.
    """

    def __init__(
        self,
        start: Point = (0.0, 0.0, 0.0),
        *,
        tick: float = 0.01,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.at: Point = start
        self.force = 0.0
        self.tick = tick
        self.log: list[Command] = []
        self.moving = False
        self.halted_at: float | None = None
        self._clock = clock
        self._stop = asyncio.Event()

    def position(self) -> Point:
        return self.at

    async def execute(self, command: Command) -> None:
        self.log.append(command)
        if command.kind == "grip":
            self.force = command.force
            return
        assert command.target is not None
        self._stop.clear()
        self.moving, self.halted_at = True, None
        try:
            await self._move(command.target, command.speed)
        finally:
            self.moving, self.halted_at = False, self._clock()

    async def _move(self, target: Point, speed: float) -> None:
        # Step by the time that really passed: OS timers (about 16 ms on Windows) oversleep.
        last = self._clock()
        while True:
            now = self._clock()
            step, last = speed * (now - last), now
            gap = math.dist(self.at, target)
            if gap <= step:
                self.at = target
                return
            f = step / gap
            self.at = tuple(a + (b - a) * f for a, b in zip(self.at, target, strict=True))  # type: ignore[assignment]
            try:
                await asyncio.wait_for(self._stop.wait(), self.tick)
                return
            except TimeoutError:
                continue

    async def stop(self) -> None:
        self._stop.set()
        self.force = 0.0
