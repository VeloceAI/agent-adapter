"""RobotGate: every robot command passes the envelope, a shadow simulation and the executor."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import GuardrailBlocked, ToolError, ToolUsageError, log_failure
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ToolCall
from agent_adapter.exec.pipeline import current_attempt
from agent_adapter.robotics.kinematic import KinematicSim
from agent_adapter.robotics.ports import RobotPort, SimPort
from agent_adapter.robotics.spec import ActionSpec, Command, Verdict, judge, judge_path
from agent_adapter.robotics.watchdog import Deadman
from agent_adapter.tracing.events import emit, make_event

if TYPE_CHECKING:
    from agent_adapter.exec.executor import Executor

logger = logging.getLogger(__name__)

DIRECT_CALL = (
    "robot tools run only through the executor (executor.run, run_tool_loop, attach with "
    "executor=): a direct call would skip the approval the envelope asks for"
)

__all__ = ["RobotGate"]


class RobotGate:
    """Puts a robot behind an `ActionSpec`, a `SimPort` shadow and the executor's approvals.

    Inside the envelope a command runs. Outside it, the tool's `confirm` rule asks the
    executor's approval port; there is no other approval path. A command that breaks a hard
    limit, enters a keep-out zone in simulation, exceeds the rate, or arrives while the
    e-stop is latched or the heartbeat is lost is blocked, also when that happens while the
    command is being simulated. `estop()` never needs the executor, an approval or a free
    slot. Cancelling a running motion call stops the robot and latches the e-stop.
    """

    def __init__(
        self,
        spec: ActionSpec,
        *,
        robot: RobotPort,
        sim: SimPort | None = None,
        executor: Executor | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.spec = spec
        self.robot = robot
        self.sim = sim if sim is not None else KinematicSim()
        self._executor = executor
        self._clock = clock
        self._latched: str | None = None
        self._epoch = 0  # bumped by every estop(): a command checked before it must not run
        self._recent: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._deadman: Deadman | None = None

    # --- tools -------------------------------------------------------------------------

    def add_tools(self, registry: Registry, *, prefix: str = "") -> list[ToolSpec]:
        """Register `move_to`, `grip`, `estop` and `reset_estop` on `registry`."""
        gate = self

        async def move_to(x: float, y: float, z: float, speed: float, ctx: ToolContext) -> str:
            """Move the robot's end effector in a straight line.

            Args:
                x: Target x in metres.
                y: Target y in metres.
                z: Target z in metres.
                speed: Speed in metres per second.
            """
            return await gate._run("move", {"x": x, "y": y, "z": z, "speed": speed}, ctx)

        async def grip(force: float, ctx: ToolContext) -> str:
            """Close the gripper.

            Args:
                force: Grip force in newtons; 0 opens it.
            """
            return await gate._run("grip", {"force": force}, ctx)

        async def estop(reason: str = "") -> str:
            """Stop the robot now. Always allowed; a person must reset it before it moves again.

            Args:
                reason: Why you are stopping it.
            """
            await gate.estop(reason or "estop tool")
            return "stopped; motion stays blocked until a person resets the e-stop"

        async def reset_estop() -> str:
            """Ask a person to release the e-stop so the robot may move again."""
            if current_attempt() is None:
                raise ToolUsageError(DIRECT_CALL)
            gate.reset()
            return "e-stop released"

        # No `effects` on motion: the executor finishes a cancelled call that declares one,
        # and a cancelled move must stop instead. parallel_safe on every robot tool keeps the
        # estop tool from queueing behind a move for the run's slot; the gate's own lock runs
        # motion one command at a time.
        motion: dict[str, Any] = {"idempotent": True, "parallel_safe": True, "tags": {"robot"}}
        return [
            registry.wrap(move_to, name=f"{prefix}move_to", confirm=self._rule("move"), **motion),
            registry.wrap(grip, name=f"{prefix}grip", confirm=self._rule("grip"), **motion),
            registry.wrap(estop, name=f"{prefix}estop", effects="additive", **motion),
            registry.wrap(
                reset_estop,
                name=f"{prefix}reset_estop",
                confirm=True,
                effects="destructive",
                **motion,
            ),
        ]

    def _rule(self, kind: str) -> Callable[[ToolCall, ToolContext], bool]:
        # A pure function of the arguments, so the tool body can rely on it having asked:
        # state (e-stop, heartbeat, rate, simulation) can only ever block, never need approval.
        def confirm(call: ToolCall, ctx: ToolContext) -> bool:
            command = _command(kind, call.arguments)
            return isinstance(command, Command) and judge(self.spec, command).action == "approve"

        return confirm

    async def _run(self, kind: str, args: Mapping[str, Any], ctx: ToolContext) -> str:
        if current_attempt() is None:
            raise ToolUsageError(DIRECT_CALL)
        command = _command(kind, args)
        async with self._lock:
            epoch = self._epoch
            verdict = await self.check(command)
            ctx.emit("robot_check", kind=kind, action=verdict.action, reasons=list(verdict.reasons))
            if verdict.action == "block":
                raise GuardrailBlocked(verdict.summary, tool=f"robot.{kind}")
            assert isinstance(command, Command)  # a bad command is blocked by check()
            # check() awaited the simulation: re-check the stop state with no await before execute.
            stopped = self._stopped_since(epoch)
            if stopped is not None:
                ctx.emit("robot_check", kind=kind, action="block", reasons=[stopped])
                raise GuardrailBlocked(stopped, tool=f"robot.{kind}")
            self._recent.append(self._clock())
            try:
                await self.robot.execute(command)
            except asyncio.CancelledError:
                await asyncio.shield(self.estop(f"{kind} cancelled"))
                raise
            if self._latched is not None:
                raise ToolError(f"e-stop ({self._latched}): stopped at {_at(self.robot)}")
            ctx.emit("robot_done", kind=kind, position=list(self.robot.position()))
        return f"done: {kind} at {_at(self.robot)}"

    # --- checks ------------------------------------------------------------------------

    async def check(self, command: Command | str) -> Verdict:
        """The gate's verdict for a command now: its envelope, the gate state, its simulation."""
        if isinstance(command, str):
            return Verdict("block", (command,))
        if self._latched is not None:
            return Verdict(
                "block", (f"e-stop is latched ({self._latched}); a person must reset it",)
            )
        if self._deadman is not None and not self._deadman.alive():
            return Verdict("block", ("no operator heartbeat",))
        if self._over_rate():
            return Verdict("block", (f"over {self.spec.max_rate:g} commands per second",))
        verdict = judge(self.spec, command)
        if verdict.action == "block":
            return verdict
        start = self.robot.position()
        if command.target is not None:
            seconds = math.dist(start, command.target) / command.speed
            if seconds > self.spec.max_duration:
                limit = self.spec.max_duration
                return Verdict(
                    "block", (f"move takes {seconds:.1f} s, over max_duration {limit:g}",)
                )
        try:
            path = await self.sim.simulate(start, command, self.spec)
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "shadow simulation failed for %s", command.kind)
            return Verdict("block", (f"shadow simulation failed: {type(exc).__name__}",))
        shadow = judge_path(self.spec, start, path)
        return shadow if shadow.action == "block" else verdict

    def _stopped_since(self, epoch: int) -> str | None:
        """Why a command checked at `epoch` may no longer run, or None."""
        if self._latched is not None or self._epoch != epoch:
            return f"e-stop during the check ({self._latched or 'released since'})"
        if self._deadman is not None and not self._deadman.alive():
            return "no operator heartbeat"
        return None

    def _over_rate(self) -> bool:
        now = self._clock()
        while self._recent and now - self._recent[0] >= 1.0:
            self._recent.popleft()
        return len(self._recent) >= self.spec.max_rate

    # --- e-stop and deadman ------------------------------------------------------------

    @property
    def latched(self) -> str | None:
        """Why the e-stop is latched, or None when the robot may move."""
        return self._latched

    async def estop(self, reason: str = "estop") -> None:
        """Latch the e-stop and stop the robot. No executor, approval or lock involved."""
        self._latched = self._latched or reason
        self._epoch += 1
        started = self._clock()
        try:
            await self.robot.stop()
        finally:
            self._audit(
                "custom.robot_estop",
                reason=reason,
                position=list(self.robot.position()),
                stop_ms=round((self._clock() - started) * 1000, 3),
            )
        logger.warning("e-stop: %s", reason)

    def reset(self) -> None:
        """Release the e-stop (the `reset_estop` tool asks a person first)."""
        reason, self._latched = self._latched, None
        rearm = getattr(self.robot, "reset", None)
        if callable(rearm):
            rearm()
        if self._deadman is not None:
            self._deadman.start()
        self._audit("custom.robot_reset", reason=reason or "")

    def heartbeat(self) -> None:
        """An operator or controlling-process heartbeat for the deadman watchdog."""
        if self._deadman is not None:
            self._deadman.beat()

    @asynccontextmanager
    async def deadman(self, timeout: float) -> AsyncIterator[Deadman]:
        """Arm a deadman: no `heartbeat()` for `timeout` seconds triggers `estop()`."""
        watchdog = Deadman(timeout, lambda: self.estop("heartbeat lost"), clock=self._clock)
        self._deadman = watchdog
        watchdog.start()
        try:
            yield watchdog
        finally:
            await watchdog.close()
            self._deadman = None

    def _audit(self, type: str, **data: Any) -> None:
        if self._executor is not None:
            emit(self._executor.sink, make_event(type, **data))


def _command(kind: str, args: Mapping[str, Any]) -> Command | str:
    """The command for a tool's arguments, or why it is not a valid one."""
    try:
        if kind == "move":
            target = (args["x"], args["y"], args["z"])
            return Command(kind="move", target=target, speed=args["speed"])
        return Command(kind="grip", force=args["force"])
    except (KeyError, ValidationError) as exc:
        return f"invalid {kind} command: {type(exc).__name__}"


def _at(robot: RobotPort) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in robot.position()) + ")"
