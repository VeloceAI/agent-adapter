"""Contract kit C1-C8 for A2A: in-process `AgentExecutor`, and over JSON-RPC with the SDK client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pytest

pytest.importorskip("a2a")
pytest.importorskip("starlette")

from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.types import (
    AgentCard,
    AgentSkill,
    Message,
    Role,
    SendMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from google.protobuf.json_format import MessageToDict

from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.a2a import (
    INTEGRATION,
    TOOLS_EXTENSION,
    a2a_app,
    a2a_executor,
    agent_card,
)
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.integrations.discovery import available_integrations
from agent_adapter.runtime.ids import new_id
from tests.integrations.a2a.support import (
    BASE,
    RawClient,
    data,
    result,
    serving,
    state,
    status_data,
)
from tests.integrations.kit import contract_tests

pytestmark = pytest.mark.contract


@dataclass
class Exported:
    skill: AgentSkill  # what the integration exports
    card: AgentCard
    runner: Any  # the Starlette app, or the AgentExecutor


def _schema(card: AgentCard, name: str) -> dict[str, Any]:
    for extension in card.capabilities.extensions:
        if extension.uri == TOOLS_EXTENSION:
            schema: dict[str, Any] = MessageToDict(extension.params)["tools"][name]
            return schema
    raise AssertionError("the card has no tools extension")


class A2AHttpHarness:
    """Calls go through the SDK's own JSON-RPC client against `a2a_app` (ASGI, no sockets)."""

    name = "a2a"
    edits_supported = True  # through the executor's port
    native_call_ids = False  # the server generates task ids
    approval: Literal["native", "port"] = "port"

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Exported:
        registry = Registry()
        registry.add(spec)
        card = agent_card(registry, url=f"{BASE}/")
        return Exported(INTEGRATION.export_tool(spec), card, (registry, executor))

    def export_all(self, registry: Registry, executor: ToolRunner) -> list[AgentSkill]:
        return INTEGRATION.export_tools(registry, executor=executor)

    async def call(self, native: Exported, args: dict[str, Any], *, call_id: str) -> Task:
        registry, executor = native.runner  # one app per call: its lifespan runs once
        app = a2a_app(registry, url=f"{BASE}/", executor=executor, allowed_hosts=["agent.test"])
        async with serving(app), RawClient(app) as client:
            return await client.call(native.skill.id, args)

    def value(self, raw: Task) -> Any:
        return result(raw) if state(raw) == "completed" else status_data(raw)

    def is_error(self, raw: Any) -> bool:
        return isinstance(raw, Task) and state(raw) in ("failed", "rejected")

    async def approve_flow(
        self, native: Exported, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        return await self.call(native, args, call_id="call_kit_c6")

    def describe(self, native: Any) -> tuple[str, str, Mapping[str, Any]]:
        if isinstance(native, Exported):
            return native.skill.id, native.skill.description, _schema(native.card, native.skill.id)
        return native.id, native.description, {}

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        return spec.json_schema()


class _Queue:
    """Records the events an `AgentExecutor` publishes."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)

    def task(self) -> Task:
        task = next(e for e in self.events if isinstance(e, Task))
        out = Task()
        out.CopyFrom(task)
        for event in self.events:
            if isinstance(event, TaskStatusUpdateEvent):
                out.status.CopyFrom(event.status)
            elif isinstance(event, TaskArtifactUpdateEvent):
                out.artifacts.append(event.artifact)
        return out


class A2AInProcessHarness(A2AHttpHarness):
    """Calls `AgentExecutor.execute` directly: the task id is the kit's call id."""

    native_call_ids = True

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Exported:
        registry = Registry()
        registry.add(spec)
        card = agent_card(registry, url=f"{BASE}/")
        return Exported(
            INTEGRATION.export_tool(spec), card, a2a_executor(registry, executor=executor)
        )

    async def call(self, native: Exported, args: dict[str, Any], *, call_id: str) -> Task:
        message = Message(
            role=Role.ROLE_USER,
            message_id=new_id("msg_"),
            parts=[data({"tool": native.skill.id, "arguments": args})],
        )
        context = RequestContext(
            ServerCallContext(),
            SendMessageRequest(message=message),
            task_id=call_id,
            context_id="ctx_kit",
        )
        queue = _Queue()
        await native.runner.execute(context, queue)
        return queue.task()


def _kit(harness: Any, suffix: str, **options: Any) -> dict[str, Any]:
    fixture, *tests = contract_tests(harness, **options)
    items = {fixture.__name__: fixture}
    for test in tests:
        if test.__name__.startswith("test_c8_") and "a2a" not in available_integrations():
            test = pytest.mark.skip(reason="the a2a entry point is not registered yet")(test)
        items[f"{test.__name__}{suffix}"] = test
    return items


globals().update(_kit(A2AInProcessHarness, ""))
globals().update(_kit(A2AInProcessHarness, "_executor", runner_factory=Executor))
# Over the wire, the SDK runs `execute` in its own task: a client-side cancel or a tool's
# own CancelledError cannot reach the caller as one, so C7 runs in-process only.
globals().update(
    (name, item)
    for name, item in _kit(A2AHttpHarness, "_http", runner_factory=Executor).items()
    if not name.startswith("test_c7_")
)


def test_bind_context_reads_the_task_id():
    native = RequestContext(ServerCallContext(), task_id="task_1", context_id="ctx_1")
    assert INTEGRATION.bind_context(native).call_id == "task_1"
    assert INTEGRATION.bind_approval(object()) is None
