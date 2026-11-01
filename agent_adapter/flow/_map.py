"""The `Map` action: one tool call or function call per item, kept by index across resumes."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalPending, FlowError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ToolCall, ToolResult
from agent_adapter.exec._resume import _journal_scope, current_journal
from agent_adapter.exec.executor import Executor
from agent_adapter.flow._invoke import describe_error, invoke
from agent_adapter.flow.actions import Map
from agent_adapter.flow.graph import NodeSpec
from agent_adapter.runtime.current import use_context

logger = logging.getLogger(__name__)

ItemOutcome = tuple[str, Any]  # ("ok", value) or ("skipped", error description)
_SKIPPED = object()  # a Map item skipped (on_error="skip") before a pause


def item_journal(index: int) -> str:
    """The child journal name of a `Map(fn=...)` item."""
    return f"item:{index}"


@dataclass
class MapVisit:
    """One visit of a Map node; `items` holds finished items by index (kept on resume)."""

    flow: str
    node: NodeSpec
    executor: Executor
    state: BaseModel
    ctx: ToolContext
    items: dict[int, ItemOutcome]

    @property
    def where(self) -> str:
        return f"flow {self.flow!r} node {self.node.name!r}"

    def _error(self, message: str) -> FlowError:
        return FlowError(f"{self.where}: {message}", flow=self.flow, node=self.node.name)

    async def run(self, action: Map) -> tuple[str, list[Any]]:
        """The state key and the kept item values, in item order.

        Raises:
            FlowError: `over` is not a list, or an item failed with `on_error="fail"`.
            ApprovalPending: an item waits for approval (finished items are kept).
        """
        items = getattr(self.state, action.over)
        if not isinstance(items, list | tuple):
            raise self._error(f"Map.over {action.over!r} is {type(items).__name__}, not a list")
        into = action.into or self.node.name
        todo = [(i, item) for i, item in enumerate(items) if i not in self.items]
        if action.fn is not None:
            fresh = await self._map_fn(action.fn, todo)
        else:
            fresh = await self._map_tools(action, todo)
        by_index = dict(zip((i for i, _ in todo), fresh, strict=True))
        for index, value in by_index.items():
            if not isinstance(value, Exception):
                self.items[index] = ("ok", value)
            elif action.on_error == "skip" and not isinstance(value, ApprovalPending):
                self.items[index] = ("skipped", describe_error(value))
        values = [
            by_index[i] if i in by_index else _done_value(self.items[i]) for i in range(len(items))
        ]
        return into, self._kept(action, values)

    async def _map_tools(self, action: Map, items: list[tuple[int, Any]]) -> list[Any]:
        if not items:
            return []
        calls = []
        for index, item in items:
            name, args = self._item_call(action, item, index)
            calls.append(
                ToolCall(name=name, arguments=dict(args), id=f"{self.ctx.call_id}:{index}")
            )
        registry: Registry | None = None
        if isinstance(action.tool, ToolSpec):
            registry = Registry()
            registry.add(action.tool)
        results = await self.executor.run_many(
            calls, self.ctx, concurrency=self.node.concurrency, registry=registry
        )
        return [_result_value(result) for result in results]

    def _item_call(self, action: Map, item: Any, index: int) -> tuple[str, Mapping[str, Any]]:
        data = item.model_dump() if isinstance(item, BaseModel) else item
        if action.tool_from is not None:
            if not isinstance(data, Mapping) or not isinstance(data.get(action.tool_from), str):
                raise self._error(f"item {index} has no tool name under {action.tool_from!r}")
            name = data[action.tool_from]
            if action.args_fn is not None:
                args: Any = action.args_fn(item)
            elif action.args_from in data:
                args = data[action.args_from]
            else:
                args = {key: value for key, value in data.items() if key != action.tool_from}
        else:
            tool = action.tool
            name = tool.name if isinstance(tool, ToolSpec) else str(tool)
            args = action.args_fn(item) if action.args_fn is not None else data
        if not isinstance(args, Mapping):
            raise self._error(f"item {index} arguments are {type(args).__name__}, not a mapping")
        return name, args

    async def _map_fn(self, fn: Callable[..., Any], items: list[tuple[int, Any]]) -> list[Any]:
        journal = current_journal()

        def item_work(index: int, item: Any) -> Callable[[], Any]:
            async def work() -> Any:
                child = self.ctx.child(call_id=f"{self.ctx.call_id}:{index}")
                # each item replays only its own calls: identical items never collide
                scope = (
                    _journal_scope(journal.child(item_journal(index)))
                    if journal is not None
                    else nullcontext()
                )
                with use_context(child), scope:
                    return await invoke(fn, item, self.state, child)

            return work

        return await self.executor.parallel(
            *[item_work(i, item) for i, item in items],
            concurrency=self.node.concurrency,
            on_error="collect",
        )

    def _kept(self, action: Map, values: Sequence[Any]) -> list[Any]:
        """Item values in order; failures fail the node or are skipped, pending pauses it."""
        failures = [(i, v) for i, v in enumerate(values) if isinstance(v, Exception)]
        errors = [(i, v) for i, v in failures if not isinstance(v, ApprovalPending)]
        if errors and action.on_error == "fail":
            index, error = errors[0]
            raise self._error(f"item {index} failed: {describe_error(error)}") from error
        for index, error in errors:
            logger.warning(
                "%s: item %d failed (%s); skipped", self.where, index, describe_error(error)
            )
        pending = [v for _, v in failures if isinstance(v, ApprovalPending)]
        if pending:
            raise pending[0]
        return [v for v in values if not isinstance(v, Exception) and v is not _SKIPPED]


def _done_value(outcome: ItemOutcome) -> Any:
    """A finished item's value; a skipped one stays skipped (not logged again)."""
    status, value = outcome
    return value if status == "ok" else _SKIPPED


def _result_value(result: ToolResult) -> Any:
    """The value of an ok result, else its exception (pending keeps its call ids)."""
    if result.ok:
        return result.value
    exc = result.exception
    if isinstance(exc, ApprovalPending):
        return with_ids(exc, tool=result.name, call_id=result.call_id)
    if isinstance(exc, Exception):
        return exc
    return FlowError(f"tool {result.name!r} failed: {result.error.kind if result.error else ''}")


def with_ids(exc: ApprovalPending, *, tool: str, call_id: str) -> ApprovalPending:
    """`exc` with the tool and call id filled in (a new error caused by it, when needed)."""
    if exc.call_id and exc.tool:
        return exc
    filled = ApprovalPending(
        exc.approval_id,
        tool=exc.tool or tool,
        run_id=exc.run_id,
        call_id=exc.call_id or call_id,
        arguments=exc.arguments,
    )
    filled.__cause__ = exc
    return filled
