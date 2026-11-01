"""Shared A2A test helpers: an app served in-process and a raw JSON-RPC A2A client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_data_parts, get_message_text, new_data_part, new_text_part
from a2a.types import (
    AgentCard,
    GetTaskRequest,
    Message,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
)

from agent_adapter.integrations.a2a import CARD_PATH
from agent_adapter.runtime.ids import new_id

BASE = "http://agent.test"
STATES = TaskState


@asynccontextmanager
async def serving(app: Any) -> AsyncIterator[Any]:
    """`app` with its lifespan running."""
    async with app.router.lifespan_context(app):
        yield app


class RawClient:
    """The SDK's own client over `httpx.ASGITransport`, with fixed headers (one caller)."""

    def __init__(self, app: Any, headers: Mapping[str, str] | None = None) -> None:
        self.app = app
        self.headers = dict(headers or {})
        self._http: httpx.AsyncClient | None = None
        self._client: Any = None

    async def __aenter__(self) -> RawClient:
        self._http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url=BASE, headers=self.headers
        )
        card = await self.card()
        self._client = await create_client(
            card, ClientConfig(httpx_client=self._http, streaming=False)
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.close()
        assert self._http is not None
        await self._http.aclose()

    async def card(self) -> AgentCard:
        from a2a.client.card_resolver import parse_agent_card

        assert self._http is not None
        response = await self._http.get(CARD_PATH)
        response.raise_for_status()
        return parse_agent_card(response.json())

    async def send(self, parts: list[Any], *, task: Task | None = None, task_id: str = "") -> Task:
        """Send one user message (a follow-up when `task` or `task_id` is given)."""
        message = Message(
            role=Role.ROLE_USER,
            message_id=new_id("msg_"),
            parts=parts,
            task_id=task.id if task is not None else task_id,
            context_id=task.context_id if task is not None else "",
        )
        result = None
        async for response in self._client.send_message(SendMessageRequest(message=message)):
            result = response.task
        assert result is not None
        return result

    async def call(self, tool: str, arguments: Mapping[str, Any]) -> Task:
        return await self.send([data({"tool": tool, "arguments": dict(arguments)})])

    async def get(self, task_id: str) -> Task:
        return await self._client.get_task(GetTaskRequest(id=task_id))


def data(value: Any) -> Any:
    return new_data_part(value)


def text(value: str) -> Any:
    return new_text_part(value)


def state(task: Task) -> str:
    return TaskState.Name(task.status.state).removeprefix("TASK_STATE_").lower()


def result(task: Task) -> Any:
    """The single JSON text part of a completed task's artifacts, parsed."""
    parts = [p for a in task.artifacts for p in a.parts]
    assert len(parts) == 1 and parts[0].media_type == "application/json", parts
    return json.loads(parts[0].text)


def status_data(task: Task) -> dict[str, Any]:
    values = get_data_parts(list(task.status.message.parts))
    assert values, task
    found: dict[str, Any] = values[0]
    return found


def status_text(task: Task) -> str:
    return get_message_text(task.status.message)
