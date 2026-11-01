"""Import a remote A2A agent's skills as tools, over a guarded HTTP client."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from agent_adapter.core.errors import IntegrationError, ToolError
from agent_adapter.core.imported import spec_from_schema
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations.a2a._guard import (
    MAX_RESPONSE_BYTES,
    HostPolicy,
    HostRefused,
    ResponseTooLarge,
    guarded_transport,
)
from agent_adapter.integrations.a2a._pinned import pinned_transport
from agent_adapter.integrations.a2a._wire import (
    ARGUMENTS_KEY,
    JSON_MEDIA,
    TOOL_KEY,
    TOOLS_EXTENSION,
    clip,
    require_a2a,
    whole_numbers,
)
from agent_adapter.integrations.a2a.card import MAX_SKILLS
from agent_adapter.runtime.ids import new_id

if TYPE_CHECKING:
    from a2a.client import Client
    from a2a.types import AgentCard, Task

logger = logging.getLogger(__name__)

CARD_PATH = "/.well-known/agent-card.json"
TIMEOUT_S = 30.0  # one HTTP request
TASK_TIMEOUT_S = 300.0  # one imported call, polling included
POLL_INTERVAL_S = 1.0
TEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"message": {"type": "string", "description": "The request, in plain text."}},
    "required": ["message"],
}
_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_]")


class A2AConnection:
    """An open A2A client, the agent card it was built from, and its HTTP client."""

    def __init__(self, card: AgentCard, client: Client, http: Any) -> None:
        self.card = card
        self.client = client
        self.http = http

    async def aclose(self) -> None:
        """Close the A2A client and its HTTP client."""
        try:
            await self.client.close()
        finally:
            await self.http.aclose()

    async def __aenter__(self) -> A2AConnection:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


async def connect_a2a(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    auth: Any = None,
    allow_private: bool = False,
    trusted_hosts: Iterable[str] = (),
    allow_cross_origin: bool = False,
    timeout_s: float = TIMEOUT_S,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    http_transport: Any = None,
) -> A2AConnection:
    """Fetch the agent card at `url` + `CARD_PATH` and open a non-streaming A2A client.

    Every request (card, calls, interfaces the card names) passes the host guard: no
    private or local addresses unless `allow_private` or the host is in `trusted_hosts`.
    Sockets go only to the addresses that check saw; a custom `http_transport` (tests,
    ASGI) is checked per request but connects on its own. `headers` / `auth` (an
    `httpx.Auth`) go only to `url`'s origin unless `allow_cross_origin=True`.
    Responses are capped at `max_response_bytes`.

    Raises:
        HostRefused: `url` (or an interface host) is not allowed.
        IntegrationError: the card cannot be fetched or parsed, or names no usable interface.
        MissingDependency: `a2a-sdk` is not installed.
    """
    require_a2a()
    import httpx

    policy = HostPolicy(allow_private=allow_private, trusted_hosts=trusted_hosts)
    await policy.check(url)
    transport = guarded_transport(
        http_transport if http_transport is not None else pinned_transport(policy),
        policy,
        max_response_bytes=max_response_bytes,
        pinned=http_transport is None,
    )
    http = httpx.AsyncClient(
        transport=transport,
        headers=dict(headers or {}),
        auth=auth,
        timeout=timeout_s,
        follow_redirects=False,
    )
    try:
        card = await _fetch_card(http, url)
        card = _usable_interfaces(card, url, allow_cross_origin)
        from a2a.client import ClientConfig, create_client
        from a2a.utils.constants import TransportProtocol

        config = ClientConfig(
            httpx_client=http,
            streaming=False,
            supported_protocol_bindings=[
                TransportProtocol.JSONRPC.value,
                TransportProtocol.HTTP_JSON.value,
            ],
        )
        client = await create_client(card, config)
    except BaseException:
        await http.aclose()
        raise
    logger.info("connected to A2A agent %r (%d skills)", card.name, len(card.skills))
    return A2AConnection(card, client, http)


async def _fetch_card(http: Any, url: str) -> AgentCard:
    import httpx
    from a2a.client.card_resolver import parse_agent_card
    from google.protobuf.json_format import ParseError

    target = url.rstrip("/") + CARD_PATH
    try:
        response = await http.get(target)
        response.raise_for_status()
        return parse_agent_card(response.json())
    except (HostRefused, ResponseTooLarge):
        raise
    except (httpx.HTTPError, ValueError, ParseError) as exc:
        where = _safe_origin(url)  # the URL may carry userinfo or a token
        logger.warning("A2A agent card at %s unusable: %s", where, type(exc).__name__)
        raise IntegrationError(
            f"cannot read the A2A agent card at {where}: {type(exc).__name__}", integration="a2a"
        ) from exc


def _usable_interfaces(card: AgentCard, url: str, cross_origin: bool) -> AgentCard:
    """`card` keeping only same-origin interfaces (credentials never go elsewhere)."""
    from a2a.types import AgentCard

    origin = _origin(url)
    kept = [i for i in card.supported_interfaces if cross_origin or _origin(i.url) == origin]
    if not kept:
        raise IntegrationError(
            f"the A2A card names no interface on {origin}; pass allow_cross_origin=True to "
            "send requests (and credentials) to another origin",
            integration="a2a",
        )
    trimmed = AgentCard()
    trimmed.CopyFrom(card)
    del trimmed.supported_interfaces[:]
    trimmed.supported_interfaces.extend(kept)
    return trimmed


def _safe_origin(url: str) -> str:
    """`scheme://host[:port]` of `url`: no userinfo, path or query."""
    scheme, host, port = _origin(url)
    shown = f"[{host}]" if ":" in host else host
    default = {"http": 80, "https": 443}.get(scheme)
    return f"{scheme}://{shown}" + ("" if port in (None, default) else f":{port}")


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        port = -1
    default = {"http": 80, "https": 443}.get(parts.scheme)
    return parts.scheme, (parts.hostname or "").lower(), port if port is not None else default


async def import_a2a(
    registry: Registry,
    connection: A2AConnection,
    *,
    prefix: str = "",
    names: Iterable[str] | None = None,
    task_timeout_s: float = TASK_TIMEOUT_S,
    poll_interval_s: float = POLL_INTERVAL_S,
    **policy: Any,
) -> list[ToolSpec]:
    """Register one tool per remote skill (all, or the skill ids in `names`).

    A skill the card's `TOOLS_EXTENSION` describes takes its typed arguments and is sent
    as `{tool, arguments}`; any other skill takes `{"message": str}` sent as text. The
    tools work while `connection` stays open. A remote `input-required` is a `ToolError`:
    it never approves anything here.

    Args:
        **policy: Options as for `build_spec`; they override `kind="a2a"`, `tags={"a2a"}`.

    Raises:
        IntegrationError: a requested skill is missing, or the card has too many skills.
        ToolDefinitionError, DuplicateTool, SchemaNotPortable: as for `registry.add_many`.
    """
    skills = list(connection.card.skills)
    if len(skills) > MAX_SKILLS:
        raise IntegrationError(f"the A2A card has over {MAX_SKILLS} skills", integration="a2a")
    wanted = None if names is None else ({names} if isinstance(names, str) else set(names))
    if wanted is not None:
        missing = sorted(wanted - {skill.id for skill in skills})
        if missing:
            raise IntegrationError(f"the A2A agent has no skills {missing}", integration="a2a")
    schemas = _tool_schemas(connection.card)
    specs = [
        _imported_spec(
            connection,
            skill,
            prefix,
            schemas.get(skill.id),
            policy,
            task_timeout_s,
            poll_interval_s,
        )
        for skill in skills
        if wanted is None or skill.id in wanted
    ]
    registry.add_many(specs)
    logger.info("imported %d A2A tools: %s", len(specs), [spec.name for spec in specs])
    return specs


def _tool_schemas(card: AgentCard) -> dict[str, dict[str, Any]]:
    from google.protobuf.json_format import MessageToDict

    for extension in card.capabilities.extensions:
        if extension.uri != TOOLS_EXTENSION:
            continue
        tools = MessageToDict(extension.params).get("tools")
        if isinstance(tools, dict):
            return {
                key: whole_numbers(value)
                for key, value in tools.items()
                if isinstance(value, dict) and value.get("type") == "object"
            }
    return {}


def _imported_spec(
    connection: A2AConnection,
    skill: Any,
    prefix: str,
    schema: dict[str, Any] | None,
    policy: Mapping[str, Any],
    task_timeout_s: float,
    poll_interval_s: float,
) -> ToolSpec:
    skill_id: str = skill.id
    typed = schema is not None

    async def invoke(**arguments: Any) -> Any:
        parts = _request_parts(skill_id, arguments, typed)
        return await call_remote(
            connection, skill_id, parts, timeout_s=task_timeout_s, poll_interval_s=poll_interval_s
        )

    options: dict[str, Any] = {"kind": "a2a", "tags": frozenset({"a2a"}), **policy}
    return spec_from_schema(
        _INVALID_NAME_CHARS.sub("_", prefix + skill_id),
        skill.description or skill.name or skill_id,
        schema if schema is not None else TEXT_SCHEMA,
        invoke,
        **options,
    )


def _request_parts(skill_id: str, arguments: Mapping[str, Any], typed: bool) -> list[Any]:
    from a2a.helpers import new_data_part, new_text_part

    if typed:
        payload = {TOOL_KEY: skill_id, ARGUMENTS_KEY: dict(arguments)}
        return [new_data_part(payload, media_type=JSON_MEDIA)]
    return [new_text_part(str(arguments.get("message", "")))]


async def call_remote(
    connection: A2AConnection,
    skill_id: str,
    parts: list[Any],
    *,
    timeout_s: float = TASK_TIMEOUT_S,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> Any:
    """Send one message, poll its task to a final or interrupted state, return its result.

    Raises:
        ToolError: the remote task failed, needs input or auth, timed out, or the agent is
            unreachable or answered too much.
        HostRefused: a request went to a host the guard does not allow.
    """
    from a2a.types import Message, Role, SendMessageRequest

    message = Message(role=Role.ROLE_USER, message_id=new_id("msg_"), parts=parts)
    # The SDK may swallow the timeout's cancellation, so polling also checks the deadline.
    deadline = asyncio.get_running_loop().time() + timeout_s
    try:
        async with asyncio.timeout(timeout_s):
            task = None
            async for response in connection.client.send_message(
                SendMessageRequest(message=message)
            ):
                if response.HasField("message"):
                    return _value(list(response.message.parts))
                task = response.task
            if task is None:
                raise ToolError("the A2A agent sent no task", tool=skill_id)
            task = await _poll(connection.client, task, poll_interval_s, deadline)
    except TimeoutError as exc:
        raise ToolError(f"the A2A task did not finish in {timeout_s:g}s", tool=skill_id) from exc
    except (HostRefused, ToolError):
        raise
    except ResponseTooLarge as exc:
        raise ToolError(str(exc), tool=skill_id) from exc
    except Exception as exc:
        if not _is_a2a_failure(exc):
            raise
        logger.warning("A2A call to %s failed: %s", skill_id, type(exc).__name__)
        raise ToolError(f"the A2A agent failed: {type(exc).__name__}", tool=skill_id) from exc
    return _outcome(task, skill_id)


def _is_a2a_failure(exc: Exception) -> bool:
    import httpx
    from a2a.utils.errors import A2AError

    return isinstance(exc, A2AError | httpx.HTTPError)


async def _poll(client: Client, task: Task, interval: float, deadline: float) -> Task:
    from a2a.types import GetTaskRequest, TaskState

    loop = asyncio.get_running_loop()
    running = (TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING)
    while task.status.state in running:
        if loop.time() >= deadline:
            raise TimeoutError
        await asyncio.sleep(interval)
        task = await client.get_task(GetTaskRequest(id=task.id))
    return task


def _outcome(task: Task, skill_id: str) -> Any:
    from a2a.helpers import get_message_text
    from a2a.types import TaskState

    state = task.status.state
    if state == TaskState.TASK_STATE_COMPLETED:
        return _value([part for artifact in task.artifacts for part in artifact.parts])
    text = clip(get_message_text(task.status.message)) if task.status.HasField("message") else ""
    label = TaskState.Name(state).removeprefix("TASK_STATE_").lower().replace("_", "-")
    logger.info("A2A task %s for %s ended %s", task.id, skill_id, label)
    raise ToolError(
        f"the A2A agent's task is {label}" + (f": {text}" if text else ""),
        tool=skill_id,
        details={"task_id": task.id, "context_id": task.context_id, "state": label},
    )


def _value(parts: list[Any]) -> Any:
    """One part's value (JSON text parsed, data with whole numbers as ints), else a list."""
    values = [_part_value(part) for part in parts if part.HasField("text") or part.HasField("data")]
    if len(values) == 1:
        return values[0]
    if all(isinstance(value, str) for value in values):
        return "\n".join(values)
    return values


def _part_value(part: Any) -> Any:
    from google.protobuf.json_format import MessageToDict

    if part.HasField("data"):
        return whole_numbers(MessageToDict(part.data))
    if part.media_type == JSON_MEDIA:
        try:
            return json.loads(part.text)
        except ValueError:
            logger.debug("an A2A JSON text part did not parse; kept as text")
    text: str = part.text
    return text
