"""Every error Agent Adapter raises; all subclass `AdapterError`.

Each error has a stable `kind` id and a `retryable` flag (another attempt may succeed).

| Area | Errors |
|---|---|
| Defining tools | ToolDefinitionError, DuplicateTool, SchemaNotPortable, MissingDependency |
| Calling tools | ToolNotFound, ToolArgsInvalid, ToolUsageError, ToolError, ToolTimeout |
| Tool crashes | ToolFailed (raised `from` the tool's own exception) |
| Run guards | RunLimitExceeded, GuardrailBlocked, BudgetExceeded, PlanInvalid |
| Approval | ApprovalDenied, ApprovalPending, ApprovalNotFound, ApprovalAlreadyDecided |
| Approval stores | ApprovalStoreCorrupt (a durable store read a corrupt row; fails closed) |
| Flows | FlowError > FlowDefinitionError > FlowCompileError; FlowError > FlowLoopExceeded |
| Models | ModelError > ModelRateLimited, ModelUnavailable, ModelTimeout, ModelAuthError |
| Models | ModelError > ModelBadRequest > ModelContextOverflow (too many tokens) |
| Models | ModelError > ModelContentFiltered, ModelOutputInvalid |
| Setup | ConfigError, IntegrationError, OptimiserError |
| HTTP server | AuthError > Unauthorized, Forbidden |

`ErrorInfo` is the JSON-safe form of a failure that a `ToolResult` carries.
"""

from agent_adapter.core.errors import (
    AdapterError,
    ApprovalAlreadyDecided,
    ApprovalDenied,
    ApprovalNotFound,
    ApprovalPending,
    ApprovalStoreCorrupt,
    AuthError,
    BudgetExceeded,
    ConfigError,
    DuplicateTool,
    FlowCompileError,
    FlowDefinitionError,
    FlowError,
    FlowLoopExceeded,
    Forbidden,
    GuardrailBlocked,
    IntegrationError,
    MissingDependency,
    ModelAuthError,
    ModelBadRequest,
    ModelContentFiltered,
    ModelContextOverflow,
    ModelError,
    ModelOutputInvalid,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
    OptimiserError,
    PlanInvalid,
    RunLimitExceeded,
    SchemaNotPortable,
    ToolArgsInvalid,
    ToolDefinitionError,
    ToolError,
    ToolFailed,
    ToolNotFound,
    ToolTimeout,
    ToolUsageError,
    Unauthorized,
)
from agent_adapter.core.types import ErrorInfo

__all__ = [
    "AdapterError",
    "ApprovalAlreadyDecided",
    "ApprovalDenied",
    "ApprovalNotFound",
    "ApprovalPending",
    "ApprovalStoreCorrupt",
    "AuthError",
    "BudgetExceeded",
    "ConfigError",
    "DuplicateTool",
    "ErrorInfo",
    "FlowCompileError",
    "FlowDefinitionError",
    "FlowError",
    "FlowLoopExceeded",
    "Forbidden",
    "GuardrailBlocked",
    "IntegrationError",
    "MissingDependency",
    "ModelAuthError",
    "ModelBadRequest",
    "ModelContentFiltered",
    "ModelContextOverflow",
    "ModelError",
    "ModelOutputInvalid",
    "ModelRateLimited",
    "ModelTimeout",
    "ModelUnavailable",
    "OptimiserError",
    "PlanInvalid",
    "RunLimitExceeded",
    "SchemaNotPortable",
    "ToolArgsInvalid",
    "ToolDefinitionError",
    "ToolError",
    "ToolFailed",
    "ToolNotFound",
    "ToolTimeout",
    "ToolUsageError",
    "Unauthorized",
]
