"""What every route handler shares: the service's parts, auth checks and response builders."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.approval.store import ApprovalRecord, ApprovalStore
from agent_adapter.core.errors import Forbidden, ToolNotFound
from agent_adapter.core.ports import LedgerPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec.control import is_control_flow
from agent_adapter.exec.executor import Executor
from agent_adapter.server._idempotency import Replays
from agent_adapter.server._payloads import refuse
from agent_adapter.server._runs import RunOwners
from agent_adapter.server._sse import Streams
from agent_adapter.server.auth import (
    APPROVER_SCOPE,
    Authorizer,
    DecisionAuthorizer,
    Principal,
    is_invoker,
    principal_ref,
)
from agent_adapter.server.events import EventStream
from agent_adapter.server.openapi import TARGET
from agent_adapter.server.problems import PROBLEM_MEDIA_TYPE, TYPE_PREFIX, Problem

logger = logging.getLogger(__name__)

JSON = "application/json"
RETRY_AFTER = {"budget-exceeded": "60", "too-many-streams": "5", "service-unavailable": "5"}


@dataclass(frozen=True)
class ServiceResponse:
    """A framework-neutral response: JSON `body`, or an SSE `stream` of bytes."""

    status: int
    body: Any = None
    media_type: str = JSON
    headers: Mapping[str, str] = field(default_factory=dict)
    stream: AsyncIterator[bytes] | None = None


def json_response(
    status: int, body: Any, headers: Mapping[str, str] | None = None
) -> ServiceResponse:
    return ServiceResponse(status=status, body=body, headers=dict(headers or {}))


def problem_response(found: Problem, headers: Mapping[str, str] | None = None) -> ServiceResponse:
    extra = dict(headers or {})
    if found.status == 401:
        extra.setdefault("WWW-Authenticate", "Bearer")
    kind = found.type.removeprefix(TYPE_PREFIX)
    if kind in RETRY_AFTER:
        extra.setdefault("Retry-After", RETRY_AFTER[kind])
    return ServiceResponse(
        status=found.status,
        body=found.model_dump(mode="json"),
        media_type=PROBLEM_MEDIA_TYPE,
        headers=extra,
    )


def encode_body(response: ServiceResponse) -> bytes:
    """The response body as UTF-8 JSON (b"" for no body).

    Raises:
        ValueError: the body holds NaN, infinity or a non-JSON value.
    """
    if response.body is None:
        return b""
    text = json.dumps(response.body, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return text.encode("utf-8")


@dataclass
class State:
    """The parts of one `ToolService`, shared by its route handlers."""

    registry: Registry
    executor: Executor
    store: ApprovalStore
    events: EventStream
    ledger: LedgerPort | None
    idempotency: Replays
    owners: RunOwners
    authorize: Authorizer
    authorize_decision: DecisionAuthorizer
    anonymous: bool
    max_listed: int = 100
    streams: Streams = field(default_factory=Streams)

    def tools(self) -> Registry:
        """The live registry view the HTTP surface exposes (`for_target("http")`)."""
        return self.registry.for_target(TARGET)

    def spec(self, name: str) -> ToolSpec:
        """The exported tool `name`.

        Raises:
            ToolNotFound: unknown, or not exported to `http`.
        """
        tools = self.tools()
        if name not in tools:
            raise ToolNotFound(f"no tool named {name!r}")
        return tools.get(name)

    async def allowed(
        self, principal: Principal, spec: ToolSpec, arguments: Mapping[str, Any] | None
    ) -> bool:
        """`authorize(...)`; only a literal True allows (a raising hook refuses with 500)."""
        try:
            verdict = await self.authorize(principal, spec, arguments)
        except Forbidden:
            return False
        except Exception as exc:
            if is_control_flow(exc):
                raise
            logger.error("authorize hook raised %s for tool %s", type(exc).__name__, spec.name)
            refuse("internal-error", "authorization failed")
        return verdict is True

    async def may_decide(self, principal: Principal, record: ApprovalRecord) -> bool:
        """`authorize_decision(...)`; only a literal True allows."""
        try:
            verdict = await self.authorize_decision(principal, record)
        except Forbidden:
            return False
        except Exception as exc:
            if is_control_flow(exc):
                raise
            logger.error(
                "authorize_decision hook raised %s for approval %s",
                type(exc).__name__,
                record.approval_id,
            )
            refuse("internal-error", "authorization failed")
        return verdict is True


def can_read(principal: Principal, record: ApprovalRecord) -> bool:
    """The invoker of the call, or a holder of `APPROVER_SCOPE`, may see an approval."""
    return APPROVER_SCOPE in principal.scopes or is_invoker(principal, record)


def forbid_other_owner(state: State, principal: Principal, run_key: str) -> None:
    """Refuse (403) when another principal owns `run_key`."""
    owner = state.owners.owner(run_key)
    if owner is not None and owner != principal.id:
        logger.warning("principal %s refused: run owned by another", principal_ref(principal.id))
        refuse("forbidden", "this run belongs to another principal")
