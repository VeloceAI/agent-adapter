"""`connect_a2a` / `import_a2a`: typed and text skills, remote states, SSRF and size guards."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("a2a")
pytest.importorskip("starlette")

import httpx
from a2a.helpers import new_data_part, new_text_part
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Task,
    TaskState,
    TaskStatus,
)
from starlette.applications import Starlette

from agent_adapter.core.errors import IntegrationError, ToolError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import build_spec
from agent_adapter.exec import Executor
from agent_adapter.integrations.a2a import (
    HostRefused,
    a2a_app,
    connect_a2a,
    import_a2a,
    static_bearer,
)
from agent_adapter.integrations.a2a.client import _poll
from tests.integrations.a2a.support import serving

BASE = "http://agent.test"


class Remote(AgentExecutor):
    """A foreign A2A agent: behaviour picked by the request text."""

    async def execute(self, context: Any, event_queue: Any) -> None:
        said = context.get_user_input()
        if said == "message":
            await event_queue.enqueue_event(
                TaskUpdater(event_queue, "", "").new_agent_message([new_text_part("hi")])
            )
            return
        task = Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
        )
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        reply = updater.new_agent_message
        if said == "fail":
            await updater.failed(reply([new_text_part("disk full")]))
        elif said == "ask":
            await updater.requires_input(reply([new_text_part("which account?")]))
        elif said == "slow":
            await asyncio.sleep(1.5)
        elif said == "big":
            await updater.add_artifact([new_text_part("x" * 20_000)])
            await updater.complete()
        else:
            await updater.add_artifact([new_text_part(f"echo {said}"), new_data_part({"n": 2})])
            await updater.complete()

    async def cancel(self, context: Any, event_queue: Any) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def _remote(interface_url: str = f"{BASE}/") -> Starlette:
    card = AgentCard(
        name="remote",
        description="A foreign agent.",
        version="1",
        supported_interfaces=[
            AgentInterface(url=interface_url, protocol_binding="JSONRPC", protocol_version="1.0")
        ],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id="chat", name="chat", description="Talk.", tags=["chat"])],
    )
    handler = DefaultRequestHandler(
        agent_executor=Remote(), task_store=InMemoryTaskStore(), agent_card=card
    )
    return Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")])


async def _connect(app: Any, **options: Any) -> Any:
    options.setdefault("trusted_hosts", ["agent.test"])
    return await connect_a2a(BASE, http_transport=httpx.ASGITransport(app=app), **options)


def _ours() -> Any:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    registry = Registry()
    registry.add(build_spec(add))
    return a2a_app(
        registry,
        url=f"{BASE}/",
        executor=Executor(),
        authenticate=static_bearer({"tok": "alice"}),
        allowed_hosts=["agent.test"],
    )


async def test_typed_skills_round_trip_with_credentials():
    async with serving(_ours()) as app:
        connection = await _connect(app, headers={"Authorization": "Bearer tok"})
        async with connection:
            local = Registry()
            [spec] = await import_a2a(local, connection, prefix="remote.", tags={"x"})
            assert spec.name == "remote_add" and spec.tags == frozenset({"x"})
            assert set(spec.json_schema()["properties"]) == {"a", "b"}
            assert await Executor().run_spec(spec, {"a": 2, "b": 40}) == 42


async def test_wrong_credentials_fail_the_card_fetch():
    async with serving(_ours()) as app:
        with pytest.raises(IntegrationError, match="agent card"):
            await _connect(app, headers={"Authorization": "Bearer wrong"})


async def test_text_skills_and_remote_states():
    async with serving(_remote()) as app, await _connect(app) as connection:
        [spec] = await import_a2a(Registry(), connection, names="chat", task_timeout_s=0.5)
        assert spec.kind == "a2a" and spec.json_schema()["required"] == ["message"]
        run = Executor().run_spec
        assert await run(spec, {"message": "hello"}) == ["echo hello", {"n": 2}]
        assert await run(spec, {"message": "message"}) == "hi"
        with pytest.raises(ToolError, match="failed: disk full") as failed:
            await run(spec, {"message": "fail"})
        assert failed.value.details["state"] == "failed"
        with pytest.raises(ToolError, match="input-required: which account"):
            await run(spec, {"message": "ask"})
        with pytest.raises(ToolError, match="did not finish"):
            await run(spec, {"message": "slow"})


async def test_unknown_skill_names_are_refused():
    async with serving(_remote()) as app, await _connect(app) as connection:
        with pytest.raises(IntegrationError, match="no skills"):
            await import_a2a(Registry(), connection, names=["chat", "nope"])


async def test_responses_over_the_limit_are_errors():
    async with serving(_remote()) as app, await _connect(app, max_response_bytes=8000) as conn:
        [spec] = await import_a2a(Registry(), conn)
        with pytest.raises(ToolError, match="over 8000 bytes"):
            await Executor().run_spec(spec, {"message": "big"})


async def test_cross_origin_interfaces_are_dropped():
    async with serving(_remote("http://other.test/")) as app:
        with pytest.raises(IntegrationError, match="no interface"):
            await _connect(app, headers={"Authorization": "Bearer tok"})
        connection = await _connect(app, allow_cross_origin=True, trusted_hosts=["agent.test"])
        async with connection:
            [spec] = await import_a2a(Registry(), connection)
            with pytest.raises(HostRefused):  # other.test is neither trusted nor resolvable here
                await Executor().run_spec(spec, {"message": "hello"})


async def test_private_hosts_are_refused_before_any_request():
    for url in ("http://127.0.0.1:8000", "http://169.254.169.254", "http://[::1]"):
        with pytest.raises(HostRefused):
            await connect_a2a(url)


async def test_missing_card_is_an_integration_error():
    app = Starlette()
    with pytest.raises(IntegrationError, match="agent card"):
        await _connect(app)


async def test_poll_waits_for_a_final_state():
    states = [TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_COMPLETED]

    class Client:
        async def get_task(self, request: Any) -> Task:
            return Task(id=request.id, status=TaskStatus(state=states.pop(0)))

    first = Task(id="t1", status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED))
    done = await _poll(Client(), first, 0, asyncio.get_running_loop().time() + 5)  # type: ignore[arg-type]
    assert done.status.state == TaskState.TASK_STATE_COMPLETED and states == []
