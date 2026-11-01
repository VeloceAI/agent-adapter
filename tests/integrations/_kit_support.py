"""Contract-kit support: the probe tools the checks export and `FakeRunner` (re-exported by kit)."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import pydantic

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalDenied, ToolTimeout
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest, ToolCall, ToolResult
from agent_adapter.hooks.base import Hook, ToolInvocation
from agent_adapter.hooks.dispatch import run_after, run_before
from agent_adapter.runtime.ids import new_id

WAIT_LIMIT_S = 10.0  # upper bound on any wait in the kit, so a broken bridge cannot hang
POLL_S = 0.005


class FakeRunner:
    """A small `ToolRunner`: validation, before/after hooks, confirm + approval, timeout.

    Approval mirrors the executor: `ctx.approval` when set, else the port, else denied.
    """

    def __init__(self, *, hooks: Sequence[Hook] = (), approval: ApprovalPort | None = None) -> None:
        self.hooks = tuple(hooks)
        self.approval = approval
        self.calls: list[ToolInvocation] = []  # calls that reached the tool

    async def run_spec(
        self,
        spec: ToolSpec,
        args: Mapping[str, Any] | None = None,
        ctx: ToolContext | None = None,
        *,
        call_id: str | None = None,
    ) -> Any:
        """Run `spec` once; raises what the executor would for these paths."""
        base = ctx if ctx is not None else ToolContext()
        child = dataclasses.replace(
            base,
            run_id=base.run_id or new_id("run_"),
            call_id=call_id or base.call_id or new_id("call_"),
        )
        validated = spec.validate_args(args)
        inv = ToolInvocation(spec, _tool_call(spec, validated, child.call_id), child)
        short = await run_before(self.hooks, "before_tool", inv)
        if short is not None:  # contract §8.3: skip approval and the call, still run after_tool
            result = ToolResult(child.call_id, spec.name, "ok", value=short.value, attempts=0)
        else:
            validated = await self._approve(inv, validated)
            self.calls.append(inv)
            value = await self._call(spec, validated, child)
            result = ToolResult(child.call_id, spec.name, "ok", value=value, attempts=1)
        return (await run_after(self.hooks, "after_tool", inv, result)).value

    async def _call(self, spec: ToolSpec, validated: pydantic.BaseModel, ctx: ToolContext) -> Any:
        timer = asyncio.timeout(spec.timeout)
        try:
            async with timer:
                return await spec.call(validated, ctx)
        except TimeoutError as exc:
            if timer.expired() and spec.timeout is not None:
                raise ToolTimeout(spec.name, spec.timeout, call_id=ctx.call_id) from exc
            raise

    async def _approve(self, inv: ToolInvocation, validated: pydantic.BaseModel) -> Any:
        spec, ctx = inv.spec, inv.ctx
        if not await _confirm(spec, inv.call, ctx):
            return validated
        decision = ctx.approval
        if decision is None and self.approval is not None:
            decision = await self.approval.request(
                ApprovalRequest(
                    approval_id=new_id("apr_"),
                    run_id=ctx.run_id,
                    call_id=ctx.call_id,
                    tool=spec.name,
                    arguments=dict(inv.call.arguments),
                )
            )
        if decision is None:
            decision = ApprovalDecision(approved=False, reason="no approval port configured")
        if not decision.approved:
            raise ApprovalDenied(spec.name, call_id=ctx.call_id, reason=decision.reason)
        if decision.edited_args is not None:
            return spec.validate_args(decision.edited_args)
        return validated


def _tool_call(spec: ToolSpec, validated: pydantic.BaseModel, call_id: str) -> ToolCall:
    arguments = validated.model_dump(mode="json", by_alias=True)
    return ToolCall(name=spec.name, arguments=arguments, id=call_id)


async def _confirm(spec: ToolSpec, call: ToolCall, ctx: ToolContext) -> bool:
    rule = spec.confirm
    if isinstance(rule, bool):
        return rule
    answer = rule(call, ctx)
    return bool(await answer) if inspect.isawaitable(answer) else bool(answer)


class Probe:
    """The tools the checks export, recording what actually executed.

    `started`, `released` and `cancelled` are thread-safe, so bridges that run the tool on
    another thread's event loop (sync frameworks via `run_sync`) work too.
    """

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.started = threading.Event()
        self.released = threading.Event()  # set by the kit to let a still-running wait end
        self.cancelled = threading.Event()  # set when the wait tool saw CancelledError
        self.add = build_spec(self._add, name="kit_add")
        self.profile = build_spec(self._profile, name="kit_profile")
        self.call_id = build_spec(self._call_id, name="kit_call_id")
        self.remove = build_spec(self._remove, name="kit_remove", confirm=True)
        self.slow = build_spec(self._wait, name="kit_slow", timeout=0.01)
        self.wait = build_spec(self._wait, name="kit_wait")
        self.cancel = build_spec(self._cancel, name="kit_cancel")

    def specs(self) -> list[ToolSpec]:
        return [self.add, self.profile, self.call_id, self.remove, self.wait]

    async def _add(self, a: int, b: int = 2) -> int:
        """Add two integers.

        Args:
            a: First addend.
            b: Second addend.
        """
        self.executed.append(("kit_add", {"a": a, "b": b}))
        return a + b

    async def _profile(self, name: str, tags: list[str]) -> dict[str, Any]:
        """Describe a person by name and tags."""
        self.executed.append(("kit_profile", {"name": name, "tags": tags}))
        return {"name": name, "tags": tags, "count": len(tags)}

    async def _call_id(self, tag: str, ctx: ToolContext) -> str:
        """Echo a tag with the call id the tool sees."""
        self.executed.append(("kit_call_id", {"tag": tag}))
        return f"{tag}:{ctx.call_id}"

    async def _remove(self, item: str) -> str:
        """Remove an item (needs approval)."""
        self.executed.append(("kit_remove", {"item": item}))
        return f"removed {item}"

    async def _wait(self, label: str) -> str:
        """Wait until cancelled or released (at most WAIT_LIMIT_S)."""
        self.executed.append(("kit_wait", {"label": label}))
        self.started.set()
        try:
            async with asyncio.timeout(WAIT_LIMIT_S):
                while not self.released.is_set():
                    await asyncio.sleep(POLL_S)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return label

    async def _cancel(self, label: str) -> str:
        """Raise CancelledError from inside the tool."""
        raise asyncio.CancelledError(label)


async def wait_until(event: threading.Event, timeout: float = WAIT_LIMIT_S) -> bool:
    """Poll a thread-safe event from async code; False on timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(POLL_S)
    return True
