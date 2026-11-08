"""Error types raised by Agent Adapter; all subclass AdapterError."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# The `X as X` imports re-export names whose code lives in the split-out base modules.
from agent_adapter.core.error_base import AdapterError as AdapterError
from agent_adapter.core.error_base import _restore
from agent_adapter.core.error_base import log_failure as log_failure
from agent_adapter.core.error_base import qualified_name as qualified_name
from agent_adapter.core.model_errors import ModelAuthError as ModelAuthError
from agent_adapter.core.model_errors import ModelBadRequest as ModelBadRequest
from agent_adapter.core.model_errors import ModelContentFiltered as ModelContentFiltered
from agent_adapter.core.model_errors import ModelContextOverflow as ModelContextOverflow
from agent_adapter.core.model_errors import ModelError as ModelError
from agent_adapter.core.model_errors import ModelOutputInvalid as ModelOutputInvalid
from agent_adapter.core.model_errors import ModelRateLimited as ModelRateLimited
from agent_adapter.core.model_errors import ModelTimeout as ModelTimeout
from agent_adapter.core.model_errors import ModelUnavailable as ModelUnavailable


class ToolDefinitionError(AdapterError):
    """A function cannot be turned into a tool."""

    kind = "tool-definition-invalid"


class DuplicateTool(AdapterError):
    """A tool with this name is already registered."""

    kind = "duplicate-tool"


class ToolNotFound(AdapterError):
    """No tool is registered under the requested name."""

    kind = "tool-not-found"


class ToolArgsInvalid(AdapterError):
    """Arguments supplied for a tool failed validation.

    `errors` holds JSON-safe dicts with `type`, `loc`, `msg` and stringified `ctx`. Input
    values are never kept, so the error is safe to log or send back to the model.
    """

    kind = "tool-args-invalid"

    def __init__(self, tool: str, errors: Iterable[Mapping[str, Any]]) -> None:
        self.tool = tool
        self.errors = [_clean_error(error) for error in errors]
        detail = "; ".join(f"{_loc_text(e['loc'])}: {e['msg']}" for e in self.errors)
        super().__init__(f"invalid arguments for tool {tool!r}: {detail}")


class ToolUsageError(AdapterError, TypeError):
    """Host code called the tool incorrectly (not the model's fault)."""

    kind = "tool-usage-error"


class SchemaNotPortable(AdapterError):
    """A schema uses features that some providers reject."""

    kind = "schema-not-portable"

    def __init__(self, problems: list[str], tool: str | None = None) -> None:
        self.problems = problems
        self.tool = tool
        where = f" for tool {tool!r}" if tool else ""
        super().__init__(f"schema{where} is not portable: " + "; ".join(problems))


class MissingDependency(AdapterError, ImportError):
    """An optional extra needed for this call is not installed."""

    kind = "missing-dependency"


class ToolError(AdapterError):
    """Raised by tool code to report a failure; the message is shown to the model."""

    kind = "tool-error"

    def __init__(
        self, message: str, *, tool: str = "", details: Mapping[str, Any] | None = None
    ) -> None:
        self.tool = tool
        self.details: dict[str, Any] = dict(details or {})
        super().__init__(message)


class ToolTimeout(AdapterError, TimeoutError):
    """A tool call ran past its timeout."""

    kind = "tool-timeout"
    retryable = True

    def __init__(self, tool: str, timeout: float, *, call_id: str = "") -> None:
        self.tool = tool
        self.timeout = timeout
        self.call_id = call_id
        super().__init__(f"tool {tool!r} timed out after {timeout:g}s")


class ToolFailed(AdapterError):
    """A tool raised a non-adapter exception; raised `from` that cause."""

    kind = "tool-failed"

    def __init__(self, tool: str, *, call_id: str = "", attempts: int = 1) -> None:
        self.tool = tool
        self.call_id = call_id
        self.attempts = attempts
        noun = "attempt" if attempts == 1 else "attempts"
        super().__init__(f"tool {tool!r} failed after {attempts} {noun}")

    @property
    def cause_type(self) -> str:
        """Qualified class name of `__cause__`; "" when there is none."""
        cause = self.__cause__
        return qualified_name(type(cause)) if cause is not None else ""


class RunLimitExceeded(AdapterError):
    """A run guard refused the call (`limit` is "once_per_run" or "max_calls")."""

    kind = "run-limit-exceeded"

    def __init__(self, tool: str, limit: str, *, run_id: str = "") -> None:
        self.tool = tool
        self.limit = limit
        self.run_id = run_id
        super().__init__(f"tool {tool!r} exceeded its run limit {limit!r}")


class GuardrailBlocked(AdapterError):
    """A guardrail rejected a payload (`stage`: args, result, llm_request, llm_response)."""

    kind = "guardrail-blocked"

    def __init__(self, reason: str, *, tool: str = "", stage: str = "args") -> None:
        self.reason = reason
        self.tool = tool
        self.stage = stage
        where = f" for tool {tool!r}" if tool else ""
        super().__init__(f"guardrail blocked {stage}{where}: {reason}")


class ApprovalDenied(AdapterError):
    """A reviewer denied a tool call."""

    kind = "approval-denied"

    def __init__(self, tool: str, *, call_id: str = "", reason: str = "") -> None:
        self.tool = tool
        self.call_id = call_id
        self.reason = reason
        suffix = f": {reason}" if reason else ""
        super().__init__(f"approval denied for tool {tool!r}{suffix}")


class ApprovalPending(AdapterError):
    """A tool call waits for a decision that arrives later (deferred approval)."""

    kind = "approval-pending"

    def __init__(
        self,
        approval_id: str,
        *,
        tool: str = "",
        run_id: str = "",
        call_id: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> None:
        self.approval_id = approval_id
        self.tool = tool
        self.run_id = run_id
        self.call_id = call_id
        self.arguments = arguments  # the requested arguments; never part of the message
        where = f" for tool {tool!r}" if tool else ""
        super().__init__(f"approval {approval_id!r} is pending{where}")

    def __reduce__(self) -> tuple[Any, ...]:
        # Arguments stay in this process: a pickled copy has none (a resume then re-asks);
        # `_aa_*` keys are process-local bookkeeping and never travel.
        state = {k: v for k, v in self.__dict__.items() if not k.startswith("_aa_")}
        state["arguments"] = None
        return (_restore, (type(self), self.args, state))


class ApprovalNotFound(AdapterError):
    """No approval request has this id."""

    kind = "approval-not-found"

    def __init__(self, approval_id: str, *, expired: bool = False, tool: str = "") -> None:
        self.approval_id = approval_id
        self.expired = expired  # a store saw its TTL run out (it was not just unknown)
        self.tool = tool
        super().__init__(f"no approval request {approval_id!r}" + (" (expired)" if expired else ""))


class ApprovalAlreadyDecided(AdapterError):
    """The approval request already has a decision."""

    kind = "approval-already-decided"

    def __init__(self, approval_id: str, *, status: str) -> None:
        self.approval_id = approval_id
        self.status = status
        super().__init__(f"approval {approval_id!r} is already {status}")


class ApprovalStoreCorrupt(AdapterError):
    """A stored approval row is unreadable or disagrees with itself; the store fails closed."""


class BudgetExceeded(AdapterError):
    """Spend reached the configured limit."""

    kind = "budget-exceeded"

    def __init__(
        self, limit_usd: float, spent_usd: float, *, scope: str = "run", run_id: str = ""
    ) -> None:
        self.limit_usd = limit_usd
        self.spent_usd = spent_usd
        self.scope = scope
        self.run_id = run_id
        super().__init__(f"{scope} budget ${limit_usd:.4f} exceeded: spent ${spent_usd:.4f}")


class PlanInvalid(AdapterError):
    """A plan of steps cannot run (unknown tools, cycles, bad references)."""

    kind = "plan-invalid"

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("invalid plan: " + "; ".join(self.problems))


class FlowError(AdapterError):
    """A flow failed while running."""

    kind = "flow-error"

    def __init__(self, message: str, *, flow: str = "", node: str = "") -> None:
        self.flow = flow
        self.node = node
        super().__init__(message)


class FlowDefinitionError(FlowError):
    """A flow definition is invalid; `problems` lists every issue found."""

    kind = "flow-definition-invalid"

    def __init__(self, problems: list[str], *, flow: str = "") -> None:
        self.problems = list(problems)
        where = f" {flow!r}" if flow else ""
        super().__init__(f"invalid flow{where}: " + "; ".join(self.problems), flow=flow)


class FlowCompileError(FlowDefinitionError):
    """A flow cannot be compiled for a target framework."""

    kind = "flow-compile-failed"

    def __init__(self, target: str, problems: list[str], *, flow: str = "") -> None:
        self.target = target
        super().__init__([f"{target}: {problem}" for problem in problems], flow=flow)
        self.problems = list(problems)


class FlowLoopExceeded(FlowError):
    """A flow node was visited more often than its loop limit allows."""

    kind = "flow-loop-exceeded"

    def __init__(self, flow: str, node: str, limit: int) -> None:
        self.limit = limit
        super().__init__(
            f"flow {flow!r}: node {node!r} exceeded its loop limit {limit}", flow=flow, node=node
        )


class CheckpointNotFound(AdapterError):
    """No checkpoint is stored for this run id."""

    kind = "checkpoint-not-found"

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"no checkpoint is stored for run {run_id!r}")


class CheckpointConflict(AdapterError):
    """Another worker saved this run's checkpoint first, or the run is not in a state to go on."""

    kind = "checkpoint-conflict"

    def __init__(self, run_id: str, message: str) -> None:
        self.run_id = run_id
        super().__init__(f"run {run_id!r}: {message}")


class CheckpointStoreCorrupt(AdapterError):
    """A stored checkpoint is unreadable, wrongly signed or disagrees with itself; fails closed."""

    kind = "checkpoint-store-corrupt"


class RecoveryNeedsDecision(AdapterError):
    """A crashed run left calls of tools not marked idempotent in doubt; a person must decide."""

    kind = "recovery-needs-decision"

    def __init__(self, run_id: str, calls: Iterable[tuple[str, str]]) -> None:
        self.run_id = run_id
        self.calls = tuple(calls)  # (tool, why) per call in doubt
        tools = ", ".join(sorted({tool for tool, _ in self.calls}))
        super().__init__(
            f"run {run_id!r}: {len(self.calls)} call(s) may or may not have run before the "
            f"crash ({tools}); check them, then recover with rerun_in_doubt=True"
        )


class ConfigError(AdapterError, ValueError):
    """Invalid configuration."""

    kind = "config-invalid"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class IntegrationError(AdapterError):
    """A framework integration failed or is unknown."""

    kind = "integration-error"

    def __init__(self, message: str, *, integration: str = "") -> None:
        self.integration = integration
        super().__init__(message)


class AuthError(AdapterError):
    """Authentication or authorisation failed."""

    kind = "auth-error"

    def __init__(self, message: str = "") -> None:
        super().__init__(message)


class Unauthorized(AuthError):
    """No valid credentials."""

    kind = "unauthorized"


class Forbidden(AuthError):
    """Valid credentials without permission."""

    kind = "forbidden"


class OptimiserError(AdapterError):
    """The optimiser could not complete."""

    kind = "optimiser-error"

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _clean_error(error: Mapping[str, Any]) -> dict[str, Any]:
    loc = [
        part if isinstance(part, int) and not isinstance(part, bool) else str(part)
        for part in error.get("loc", ())
    ]
    clean: dict[str, Any] = {
        "type": str(error.get("type", "")),
        "loc": loc,
        "msg": str(error.get("msg", "")),
    }
    ctx = error.get("ctx")
    if isinstance(ctx, Mapping) and ctx:
        clean["ctx"] = {str(key): str(value) for key, value in ctx.items()}
    return clean


def _loc_text(loc: list[str | int]) -> str:
    return ".".join(str(part) for part in loc) or "$"
