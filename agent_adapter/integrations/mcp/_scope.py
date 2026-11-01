"""Principal-scoped run and call ids; a client call key counts only to resume."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from agent_adapter.approval.store import ApprovalStore
from agent_adapter.core.errors import log_failure
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations.mcp._resume import (
    InFlight,
    ResumeKeys,
    arguments_digest,
    call_arguments,
    refuse,
    store_resumable,
)
from agent_adapter.integrations.mcp.annotations import CALL_ID_META, RUN_ID_META
from agent_adapter.runtime.ids import new_id, scoped_id

if TYPE_CHECKING:
    from mcp.server.context import ServerRequestContext

logger = logging.getLogger(__name__)

PrincipalHook = Callable[["ServerRequestContext[Any, Any]"], "str | None"]

RUN_ID_HEADER = "x-run-id"
MAX_KEY_LENGTH = 256  # client-chosen keys end up in stores and logs


@dataclass(frozen=True)
class Scope:
    """Who is calling and under which keys; `run_id` / `call_id` are the derived ids."""

    principal: str | None
    run_key: str
    call_key: str
    run_id: str
    call_id: str
    keys_honoured: bool  # the client may name runs and resume calls on this request
    in_flight: tuple[str, str] | None = None  # claimed in `InFlight`; release when done
    resuming: bool = False  # the call key resumes a pending result of this principal


@dataclass
class Resumes:
    """Where resumable calls are looked up: the shared store, else this process's keys."""

    keys: ResumeKeys
    in_flight: InFlight
    store: ApprovalStore | None = None


class _WrappedHook:
    """A principal hook that delegates to `hook`; subclasses only mark how it is trusted."""

    def __init__(self, hook: PrincipalHook) -> None:
        self.hook = hook

    def __call__(self, ctx: ServerRequestContext[Any, Any]) -> str | None:
        return self.hook(ctx)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.hook!r})"


class GatedPrincipal(_WrappedHook):
    """A hook behind a gate that already authenticated the request (the HTTP tool server)."""


class RequiredPrincipal(_WrappedHook):
    """A hook whose None principal is refused: the app has no anonymous opt-in."""


def principal_of(ctx: ServerRequestContext[Any, Any], hook: PrincipalHook | None) -> str | None:
    """The hook's principal ("" counts as none); a raising or ill-typed hook fails closed.

    Raises:
        MCPError: the hook raised or returned something other than str / None, or a
            `RequiredPrincipal` hook named no caller.
    """
    if hook is None:
        return None
    try:
        principal = hook(ctx)
    except Exception as exc:
        log_failure(logger, logging.ERROR, exc, "MCP principal hook raised; refusing the request")
        refuse("principal check failed", "INTERNAL_ERROR")
    if principal is not None and not isinstance(principal, str):
        logger.error("MCP principal hook returned %s, not str or None", type(principal).__name__)
        refuse("principal check failed", "INTERNAL_ERROR")
    if not principal and isinstance(hook, RequiredPrincipal):
        logger.warning("MCP request refused: the principal hook named no caller")
        refuse(
            "authentication required: this MCP server serves identified callers only",
            "INVALID_REQUEST",
        )
    return principal or None


async def resolve_scope(
    ctx: ServerRequestContext[Any, Any],
    meta: Mapping[str, Any],
    *,
    spec: ToolSpec,
    args: Mapping[str, Any],
    hook: PrincipalHook | None,
    allow_anonymous: bool | None,
    resumes: Resumes,
) -> Scope:
    """Keys from `_meta` / `X-Run-Id` when the caller may use them, else fresh ones.

    `allow_anonymous=None` honours anonymous keys only off HTTP (`ctx.request is None`).

    Raises:
        MCPError: a resume with other arguments, a resume already running, or a store failure.
    """
    principal = principal_of(ctx, hook)
    anonymous_ok = ctx.request is None if allow_anonymous is None else allow_anonymous
    honoured = principal is not None or anonymous_ok
    run_key, call_key = _meta_key(meta, RUN_ID_META), _meta_key(meta, CALL_ID_META)
    if not run_key:
        run_key = _header_key(ctx)
    if not honoured:
        if run_key or call_key:
            logger.warning("anonymous MCP request: ignoring the client's run and call ids")
        run_key = call_key = ""
    run_key = run_key or new_id("run_")
    run_id = scoped_id("run_", principal, run_key)
    in_flight, resuming = None, False
    if call_key:
        call_id = scoped_id("call_", principal, call_key)
        resuming, in_flight = await _resume(
            principal, run_key, call_key, run_id, call_id, spec, args, resumes
        )
        if not resuming:
            logger.debug("MCP call key is not a pending resume: minting a fresh call id")
            call_key = ""
    call_key = call_key or new_id("call_")
    return Scope(
        principal=principal,
        run_key=run_key,
        call_key=call_key,
        run_id=run_id,
        call_id=scoped_id("call_", principal, call_key),
        keys_honoured=honoured,
        in_flight=in_flight,
        resuming=resuming,
    )


async def _resume(
    principal: str | None,
    run_key: str,
    call_key: str,
    run_id: str,
    call_id: str,
    spec: ToolSpec,
    args: Mapping[str, Any],
    resumes: Resumes,
) -> tuple[bool, tuple[str, str] | None]:
    """(resuming, the in-flight claim to release); refuses rather than spend a key.

    The in-flight claim comes first and lasts the whole call, so a concurrent duplicate is
    refused even after the winner consumed the approval.
    """
    call = (run_id, call_id)
    if not resumes.in_flight.claim(call):
        logger.warning("MCP resume of %s refused: already being resumed", spec.name)
        refuse("this call is already being resumed")
    try:
        resuming = await _resumable(principal, run_key, call_key, call, spec, args, resumes)
    except BaseException:
        resumes.in_flight.release(call)
        raise
    if not resuming:
        resumes.in_flight.release(call)
        return False, None
    return True, call


async def _resumable(
    principal: str | None,
    run_key: str,
    call_key: str,
    call: tuple[str, str],
    spec: ToolSpec,
    args: Mapping[str, Any],
    resumes: Resumes,
) -> bool:
    arguments = call_arguments(spec, args)
    if resumes.store is not None:
        record = await store_resumable(resumes.store, *call, spec.name)
        if record is None:
            return False
        if record.request.arguments != arguments:
            _swapped(spec)
        return True
    outcome = resumes.keys.claim(
        principal, run_key, call_key, spec.name, arguments_digest(arguments)
    )
    if outcome == "mismatch":
        _swapped(spec)
    return outcome == "claimed"


def _swapped(spec: ToolSpec) -> NoReturn:
    logger.warning("MCP resume of %s with other arguments refused", spec.name)
    refuse("the resume arguments differ from the pending call")


def _meta_key(meta: Mapping[str, Any], name: str) -> str:
    value = meta.get(name)
    if value is None:
        return ""
    if isinstance(value, str) and 0 < len(value) <= MAX_KEY_LENGTH and value.isprintable():
        return value
    logger.warning("ignoring invalid %s in the MCP request", name)
    return ""


def _header_key(ctx: ServerRequestContext[Any, Any]) -> str:
    headers = getattr(ctx.request, "headers", None)
    if not isinstance(headers, Mapping):
        return ""
    return _meta_key({RUN_ID_META: headers.get(RUN_ID_HEADER)}, RUN_ID_META)
