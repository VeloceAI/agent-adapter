"""The `Server` callbacks: list and call registry tools, map outcomes to MCP results."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.store import ApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import AdapterError, ApprovalPending, MissingDependency, log_failure
from agent_adapter.core.registry import Registry, exported_to
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ErrorInfo
from agent_adapter.exec.control import is_control_flow
from agent_adapter.exec.defaults import runner_for
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.integrations.mcp._approval import ClientApproval
from agent_adapter.integrations.mcp._resume import (
    InFlight,
    ResumeKeys,
    arguments_digest,
    call_arguments,
)
from agent_adapter.integrations.mcp._scope import (
    PrincipalHook,
    Resumes,
    Scope,
    _meta_key,
    resolve_scope,
)
from agent_adapter.integrations.mcp.annotations import (
    APPROVAL_ID_META,
    CALL_ID_META,
    ERROR_KIND_META,
    RUN_ID_META,
    input_schema_for,
    output_schema_for,
    structured_content,
    text_content,
    tool_annotations,
)

if TYPE_CHECKING:
    import mcp.types as t
    from mcp.server.context import ServerRequestContext

    from agent_adapter.integrations.mcp._task_service import TaskService

logger = logging.getLogger(__name__)

TARGET = "mcp"


def require_mcp() -> None:
    """Raise MissingDependency when the `mcp` package is not installed."""
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        logger.debug("mcp is not importable: %s", exc)
        raise MissingDependency(
            'the MCP integration needs the mcp package: pip install "agent-adapter[mcp]"'
        ) from exc


def mcp_tool(spec: ToolSpec) -> t.Tool:
    """The MCP tool definition for `spec`: schemas, wrapped output schema, annotations."""
    require_mcp()
    import mcp.types as t

    hints = tool_annotations(spec)
    return t.Tool(
        name=spec.name,
        description=spec.description,
        input_schema=input_schema_for(spec),
        output_schema=output_schema_for(spec),
        annotations=t.ToolAnnotations(**hints) if hints else None,
    )


def raw_context(native: Any) -> ToolContext:
    """A context with the client's raw, unscoped `_meta` keys; never build a server on it.

    `mcp_server` scopes keys by principal and honours call keys only to resume.
    """
    meta = meta_of(native)
    return ToolContext(
        run_id=_meta_key(meta, RUN_ID_META),
        call_id=_meta_key(meta, CALL_ID_META),
        extras={TARGET: native},
    )


def meta_of(native: Any) -> Mapping[str, Any]:
    """The request `_meta` of `CallToolRequestParams` or a mapping; {} when absent."""
    if isinstance(native, Mapping):
        meta = native.get("_meta", native.get("meta"))
    else:
        meta = getattr(native, "meta", None)
    return meta if isinstance(meta, Mapping) else {}


class Handlers:
    """`on_list_tools`, `on_call_tool` and `get_tool_input_schema`; the registry is live."""

    def __init__(
        self,
        registry: Registry,
        executor: ToolRunner | None,
        *,
        approval: ClientApproval | None = None,
        principal: PrincipalHook | None = None,
        allow_anonymous: bool | None = None,
        tasks: TaskService | None = None,
    ) -> None:
        self.registry = registry
        self.tasks = tasks
        self.executor = executor
        self.approval = approval
        self.principal = principal
        self.allow_anonymous = allow_anonymous
        self.resume_keys = ResumeKeys()
        self.in_flight = InFlight()

    async def list_tools(
        self, ctx: ServerRequestContext[Any, Any], params: t.PaginatedRequestParams | None
    ) -> t.ListToolsResult:
        import mcp.types as t

        exported = list(self.registry.for_target(TARGET))
        visible = getattr(runner_for(self.executor), "visible", None)
        if callable(visible):  # a runner may hide tools per caller (the HTTP server's authorize)
            exported = [spec for spec in exported if await visible(spec, ctx) is True]
        return t.ListToolsResult(tools=[mcp_tool(spec) for spec in exported])

    def input_schema(self, name: str) -> dict[str, Any] | None:
        spec = self._exported(name)
        return input_schema_for(spec) if spec is not None else None

    async def call_tool(
        self, ctx: ServerRequestContext[Any, Any], params: t.CallToolRequestParams
    ) -> Any:  # CallToolResult | InputRequiredResult | a CreateTaskResult dict
        spec = self._spec(params.name)
        args = dict(params.arguments or {})
        runner = runner_for(self.executor)
        port = getattr(runner, "approval", None)
        store = port.store if isinstance(port, DeferredApproval) else None
        scope = await resolve_scope(
            ctx,
            meta_of(params),
            spec=spec,
            args=args,
            hook=self.principal,
            allow_anonymous=self.allow_anonymous,
            resumes=Resumes(self.resume_keys, self.in_flight, store),
        )
        try:
            return await self._run(ctx, params, spec, args, scope, runner, store)
        finally:
            if scope.in_flight is not None:
                self.in_flight.release(scope.in_flight)

    async def _run(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: t.CallToolRequestParams,
        spec: ToolSpec,
        args: dict[str, Any],
        scope: Scope,
        runner: ToolRunner,
        store: ApprovalStore | None,
    ) -> Any:
        tctx = ToolContext(
            run_id=scope.run_id,
            call_id=scope.call_id,
            extras={TARGET: params, "mcp_request": ctx, "principal": scope.principal},
        )
        if self.approval is not None:
            from mcp import MCPError

            try:
                gated = await self.approval.gate(
                    ctx, params, spec, args, tctx, store, resuming=scope.resuming
                )
            except MCPError:
                raise  # a bad or reused requestState
            except Exception as exc:
                if is_control_flow(exc):
                    raise
                return _error_result(spec, tctx, scope, exc)
            if not isinstance(gated, ToolContext):
                return gated
            tctx = gated
        final = tctx

        async def attempt() -> t.CallToolResult | ApprovalPending:
            return await self._attempt(spec, args, final, scope, runner)

        def settle(exc: ApprovalPending) -> t.CallToolResult:
            return self._settle(spec, args, scope, store, exc)

        if self.tasks is not None and self.tasks.accepts(ctx, scope):
            return await self.tasks.call(ctx, spec.name, scope, store, attempt, settle)
        outcome = await attempt()
        return settle(outcome) if isinstance(outcome, ApprovalPending) else outcome

    async def _attempt(
        self,
        spec: ToolSpec,
        args: dict[str, Any],
        tctx: ToolContext,
        scope: Scope,
        runner: ToolRunner,
    ) -> t.CallToolResult | ApprovalPending:
        """Run once: the tool's result or error result, or the pending approval."""
        from mcp import MCPError

        try:
            value = await runner.run_spec(spec, args, tctx, call_id=tctx.call_id)
        except ApprovalPending as exc:
            return exc
        except MCPError:
            raise  # a protocol error (e.g. a bad or reused requestState), not a tool failure
        except Exception as exc:
            if is_control_flow(exc):
                raise
            return _error_result(spec, tctx, scope, exc)
        return _ok_result(spec, value)

    def _settle(
        self,
        spec: ToolSpec,
        args: dict[str, Any],
        scope: Scope,
        store: ApprovalStore | None,
        exc: ApprovalPending,
    ) -> t.CallToolResult:
        """The `approval required` result, with resume keys the caller may use."""
        if scope.keys_honoured and store is None:  # a store vouches for its own records
            digest = arguments_digest(call_arguments(spec, args))
            self.resume_keys.issue(
                scope.principal, scope.run_key, scope.call_key, spec.name, digest
            )
        return _pending_result(spec, exc, scope)

    def _exported(self, name: str) -> ToolSpec | None:
        if name not in self.registry:
            return None
        spec = self.registry.get(name)
        return spec if exported_to(spec, TARGET) else None

    def _spec(self, name: str) -> ToolSpec:
        spec = self._exported(name)
        if spec is not None:
            return spec
        from mcp import MCPError
        from mcp.types import INVALID_PARAMS

        logger.warning("MCP call to unknown or unexported tool %r", name)
        raise MCPError(code=INVALID_PARAMS, message=f"Unknown tool: {name}")


def _ok_result(spec: ToolSpec, value: Any) -> t.CallToolResult:
    import mcp.types as t

    return t.CallToolResult(
        content=[t.TextContent(type="text", text=text_content(value))],
        structured_content=structured_content(spec, value),
    )


def _pending_result(spec: ToolSpec, exc: ApprovalPending, scope: Scope) -> t.CallToolResult:
    import mcp.types as t

    logger.info(
        "tool %s waits for approval %s (run %s, call %s)",
        spec.name,
        exc.approval_id,
        exc.run_id,
        exc.call_id,
    )
    meta: dict[str, Any] = {
        APPROVAL_ID_META: exc.approval_id,
        ERROR_KIND_META: ApprovalPending.kind,
    }
    if scope.keys_honoured:  # the keys to resume with, never the derived ids
        meta.update({RUN_ID_META: scope.run_key, CALL_ID_META: scope.call_key})
    return t.CallToolResult(
        content=[t.TextContent(type="text", text=f"approval required: {exc.approval_id}")],
        is_error=True,
        meta=meta,
    )


def _error_result(
    spec: ToolSpec, tctx: ToolContext, scope: Scope, exc: Exception
) -> t.CallToolResult:
    import mcp.types as t

    info = ErrorInfo.from_exception(exc)
    text = info.message  # derived ids never reach the client: its keys, or a placeholder
    for derived, shown in (
        (scope.run_id, scope.run_key),
        (scope.call_id, scope.call_key),
        (tctx.run_id, "<run>"),
        (tctx.call_id, "<call>"),
    ):
        text = text.replace(derived, shown)
    if isinstance(exc, AdapterError):
        logger.warning(
            "tool %s failed over MCP: %s (run %s, call %s)",
            spec.name,
            info.kind,
            tctx.run_id,
            tctx.call_id,
        )
    else:
        # Only a runner other than the Executor lets a foreign exception through; the client
        # sees its class name only, and the traceback is kept at DEBUG for the operator.
        log_failure(
            logger,
            logging.ERROR,
            exc,
            "tool %s raised over MCP (run %s, call %s)",
            spec.name,
            tctx.run_id,
            tctx.call_id,
        )
    return t.CallToolResult(
        content=[t.TextContent(type="text", text=text)],
        is_error=True,
        meta={ERROR_KIND_META: info.kind},
    )
