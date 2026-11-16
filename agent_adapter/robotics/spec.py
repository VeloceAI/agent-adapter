"""Action envelopes: what a robot command may do alone, with approval, or never."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Action",
    "ActionSpec",
    "Box",
    "Command",
    "Point",
    "Verdict",
    "Waypoint",
    "judge",
    "judge_path",
]

Point = tuple[float, float, float]
Action = Literal["run", "approve", "block"]

_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Box(BaseModel):
    """An axis-aligned box in metres, from corner `lo` to corner `hi`."""

    model_config = _STRICT

    lo: Point
    hi: Point
    name: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> Box:
        if any(a > b for a, b in zip(self.lo, self.hi, strict=True)):
            raise ValueError(f"box {self.name or ''}: lo must be <= hi on every axis")
        return self

    def contains(self, p: Point) -> bool:
        return all(a <= v <= b for a, v, b in zip(self.lo, p, self.hi, strict=True))

    def crosses(self, start: Point, end: Point) -> bool:
        """True when the straight segment from `start` to `end` touches the box (slab test)."""
        t0, t1 = 0.0, 1.0
        for a, b, lo, hi in zip(start, end, self.lo, self.hi, strict=True):
            d = b - a
            if d == 0:
                if not lo <= a <= hi:
                    return False
                continue
            near, far = sorted(((lo - a) / d, (hi - a) / d))
            t0, t1 = max(t0, near), min(t1, far)
            if t0 > t1:
                return False
        return True


class ActionSpec(BaseModel):
    """The envelope a robot works in without a person, and the hard limits beyond it.

    Inside `workspace`, `max_speed` and `max_force` a command runs. Outside them it needs an
    approval, unless it breaks a hard limit (`hard_*`, `keep_out`) or the command rate: then
    it is blocked and no approval can run it. Unset hard limits leave approval as the only
    ceiling; keep-out zones always block, also when a simulated path only passes through one.
    A speed below `min_speed` (default: a thousandth of `max_speed`) and a move that would
    take longer than `max_duration` seconds are blocked too.
    """

    model_config = _STRICT

    workspace: Box
    max_speed: float = Field(gt=0, description="m/s")
    max_force: float = Field(ge=0, description="N")
    min_speed: float | None = Field(default=None, gt=0, description="m/s")
    max_rate: float = Field(default=5.0, gt=0, description="motion commands per second")
    max_duration: float = Field(default=60.0, gt=0, description="s, one move")
    keep_out: tuple[Box, ...] = ()
    hard_workspace: Box | None = None
    hard_max_speed: float | None = Field(default=None, gt=0)
    hard_max_force: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _hard_wider(self) -> ActionSpec:
        if self.hard_max_speed is not None and self.hard_max_speed < self.max_speed:
            raise ValueError("hard_max_speed must be >= max_speed")
        if self.hard_max_force is not None and self.hard_max_force < self.max_force:
            raise ValueError("hard_max_force must be >= max_force")
        if self.min_speed is not None and self.min_speed > self.max_speed:
            raise ValueError("min_speed must be <= max_speed")
        return self

    @property
    def slowest_speed(self) -> float:
        """The lowest speed a move may use: `min_speed`, or a thousandth of `max_speed`."""
        return self.min_speed if self.min_speed is not None else self.max_speed / 1000


class Command(BaseModel):
    """One robot command: a move of the end effector, or a grip."""

    model_config = _STRICT

    kind: Literal["move", "grip"]
    target: Point | None = None
    speed: float = Field(default=0.0, ge=0)
    force: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _shape(self) -> Command:
        if self.kind == "move" and (self.target is None or self.speed <= 0):
            raise ValueError("a move needs a target and a speed above 0")
        return self


class Waypoint(BaseModel):
    """One sample of a simulated path: where the end effector is at time `t` (s)."""

    model_config = _STRICT

    t: float
    at: Point


@dataclass(frozen=True)
class Verdict:
    """What the gate does with a command, and why."""

    action: Action
    reasons: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        return "; ".join(self.reasons) or "inside the envelope"


def judge(spec: ActionSpec, command: Command) -> Verdict:
    """Judge a command on its own arguments: the part an approval can cover."""
    hard: list[str] = []
    soft: list[str] = []
    if command.kind == "move":
        assert command.target is not None  # checked by Command
        _limit(soft, hard, "speed", command.speed, spec.max_speed, spec.hard_max_speed, "m/s")
        if command.speed < spec.slowest_speed:
            hard.append(f"speed {command.speed:g} m/s is below the minimum {spec.slowest_speed:g}")
        if not spec.workspace.contains(command.target):
            soft.append(f"target {_fmt(command.target)} is outside the workspace")
        if spec.hard_workspace is not None and not spec.hard_workspace.contains(command.target):
            hard.append(f"target {_fmt(command.target)} is outside the hard workspace")
    else:
        _limit(soft, hard, "force", command.force, spec.max_force, spec.hard_max_force, "N")
    if hard:
        return Verdict("block", tuple(hard))
    return Verdict("approve", tuple(soft)) if soft else Verdict("run")


def judge_path(spec: ActionSpec, start: Point, path: Sequence[Waypoint]) -> Verdict:
    """Judge a simulated path against keep-out zones and hard limits (never approvable)."""
    reasons: list[str] = []
    points = [start, *(w.at for w in path)]
    for zone in spec.keep_out:
        if any(zone.crosses(a, b) for a, b in itertools.pairwise(points)):
            reasons.append(f"path enters keep-out zone {zone.name or _fmt(zone.lo)}")
    if spec.hard_workspace is not None and not all(map(spec.hard_workspace.contains, points)):
        reasons.append("simulated path leaves the hard workspace")
    if spec.hard_max_speed is not None:
        fastest = _fastest(start, path)
        if fastest > spec.hard_max_speed * 1.05:  # 5% slack for sampling noise
            reasons.append(f"simulated speed {fastest:.2f} m/s is over the hard limit")
    return Verdict("block", tuple(reasons)) if reasons else Verdict("run")


def _limit(
    soft: list[str],
    hard: list[str],
    what: str,
    value: float,
    cap: float,
    ceiling: float | None,
    unit: str,
) -> None:
    if ceiling is not None and value > ceiling:
        hard.append(f"{what} {value:g} {unit} is over the hard limit {ceiling:g}")
    elif value > cap:
        soft.append(f"{what} {value:g} {unit} is over the envelope {cap:g}")


def _fastest(start: Point, path: Sequence[Waypoint]) -> float:
    fastest, prev_t, prev = 0.0, 0.0, start
    for w in path:
        dt = w.t - prev_t
        if dt > 0:
            fastest = max(fastest, math.dist(prev, w.at) / dt)
        prev_t, prev = w.t, w.at
    return fastest


def _fmt(p: Point) -> str:
    return "(" + ", ".join(f"{v:g}" for v in p) + ")"
