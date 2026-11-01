"""A runner's optional `visible(spec, ctx)` hook filters `tools/list` per caller."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp import Client

from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec import Executor
from agent_adapter.integrations.mcp import mcp_server

pytestmark = pytest.mark.contract


def make_registry() -> Registry:
    reg = Registry()

    @reg.tool
    def public() -> str:
        """Shown to everyone."""
        return "p"

    @reg.tool
    def secret() -> str:
        """Hidden by the runner."""
        return "s"

    return reg


class Hiding:
    """An executor wrapper whose `visible` hides `secret`; non-True answers hide too."""

    def __init__(self, answer: Any) -> None:
        self.executor = Executor()
        self.answer = answer
        self.seen: list[Any] = []

    @property
    def approval(self) -> Any:
        return None

    async def visible(self, spec: ToolSpec, ctx: Any) -> Any:
        self.seen.append(ctx)
        return self.answer if spec.name == "secret" else True

    async def run_spec(self, spec: ToolSpec, args: Any = None, ctx: Any = None, **kw: Any) -> Any:
        return await self.executor.run_spec(spec, args, ctx, **kw)


@pytest.mark.parametrize("answer", [False, 1, None])
async def test_tools_the_runner_hides_are_not_listed(answer: Any):
    runner = Hiding(answer)
    async with Client(mcp_server(make_registry(), executor=runner)) as client:
        names = [tool.name for tool in (await client.list_tools()).tools]
    assert names == ["public"]
    assert runner.seen  # the request context reaches the hook


async def test_without_a_hook_every_tool_is_listed():
    async with Client(mcp_server(make_registry(), executor=Executor())) as client:
        names = [tool.name for tool in (await client.list_tools()).tools]
    assert names == ["public", "secret"]
