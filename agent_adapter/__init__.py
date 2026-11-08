"""Agent Adapter: framework-independent tools, parallel execution and flows for LLM agents.

Names load on first use (PEP 562), so `import agent_adapter` stays cheap. Every other
public name lives in its home subpackage: errors in `agent_adapter.errors`, the rest in
`agent_adapter.exec`, `.approval`, `.hooks`, `.cost`, `.cache`, `.flow` and so on.
"""

from agent_adapter._version import __version__

# Not `typing.TYPE_CHECKING`: importing typing would triple the import cost. Type checkers
# treat any constant named TYPE_CHECKING as true.
TYPE_CHECKING = False

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from agent_adapter.approval.builtin import AutoApprove, AutoDeny, CallbackApproval
    from agent_adapter.approval.deferred import DeferredApproval
    from agent_adapter.approval.redis_store import RedisApprovalStore
    from agent_adapter.approval.sqlite_store import SQLiteApprovalStore
    from agent_adapter.approval.store import MemoryApprovalStore
    from agent_adapter.cache.memory import MemoryCache
    from agent_adapter.core._default import default_registry, tool, wrap
    from agent_adapter.core.context import ToolContext
    from agent_adapter.core.errors import (
        AdapterError,
        ApprovalDenied,
        ApprovalPending,
        MissingDependency,
        ToolError,
    )
    from agent_adapter.core.imported import spec_from_schema
    from agent_adapter.core.messages import LLMRequest, LLMResponse, Message, ToolDef
    from agent_adapter.core.registry import Registry
    from agent_adapter.core.tool import ToolSpec, build_spec
    from agent_adapter.core.types import (
        ApprovalDecision,
        ApprovalRequest,
        Decision,
        Event,
        LedgerEntry,
        ToolCall,
        ToolResult,
        Usage,
    )
    from agent_adapter.cost.ledger import Ledger
    from agent_adapter.cost.prices import Prices
    from agent_adapter.exec.defaults import acall, call, configure, set_default_executor
    from agent_adapter.exec.executor import Executor
    from agent_adapter.exec.guards import RunLimits
    from agent_adapter.exec.plan import Step
    from agent_adapter.exec.tool_loop import run_tool_loop
    from agent_adapter.flow.runner import FlowRunner
    from agent_adapter.flow.spec import Flow
    from agent_adapter.hooks.base import Hook
    from agent_adapter.hooks.builtin.budget import BudgetHook
    from agent_adapter.hooks.builtin.guardrail import GuardrailHook
    from agent_adapter.hooks.builtin.logging import LoggingHook
    from agent_adapter.hooks.builtin.rate_limit import RateLimitHook
    from agent_adapter.integrations._attach import attach
    from agent_adapter.models.callable import CallableModel
    from agent_adapter.models.fallback import FallbackModel
    from agent_adapter.runtime.sync import run_sync

# name -> module that defines it; kept in step with the TYPE_CHECKING block by tests.
_LAZY: dict[str, str] = {
    # Tools
    "Registry": "agent_adapter.core.registry",
    "ToolSpec": "agent_adapter.core.tool",
    "build_spec": "agent_adapter.core.tool",
    "spec_from_schema": "agent_adapter.core.imported",
    "ToolContext": "agent_adapter.core.context",
    "ToolCall": "agent_adapter.core.types",
    "ToolResult": "agent_adapter.core.types",
    "Usage": "agent_adapter.core.types",
    # Running
    "Executor": "agent_adapter.exec.executor",
    "RunLimits": "agent_adapter.exec.guards",
    "Step": "agent_adapter.exec.plan",
    "run_tool_loop": "agent_adapter.exec.tool_loop",
    "run_sync": "agent_adapter.runtime.sync",
    "set_default_executor": "agent_adapter.exec.defaults",
    "configure": "agent_adapter.exec.defaults",
    "call": "agent_adapter.exec.defaults",
    "acall": "agent_adapter.exec.defaults",
    "attach": "agent_adapter.integrations._attach",
    # Approval
    "AutoApprove": "agent_adapter.approval.builtin",
    "AutoDeny": "agent_adapter.approval.builtin",
    "CallbackApproval": "agent_adapter.approval.builtin",
    "DeferredApproval": "agent_adapter.approval.deferred",
    "ApprovalRequest": "agent_adapter.core.types",
    "ApprovalDecision": "agent_adapter.core.types",
    "Decision": "agent_adapter.core.types",
    "MemoryApprovalStore": "agent_adapter.approval.store",
    "SQLiteApprovalStore": "agent_adapter.approval.sqlite_store",
    "RedisApprovalStore": "agent_adapter.approval.redis_store",
    # Hooks, cost, cache
    "Hook": "agent_adapter.hooks.base",
    "LoggingHook": "agent_adapter.hooks.builtin.logging",
    "GuardrailHook": "agent_adapter.hooks.builtin.guardrail",
    "BudgetHook": "agent_adapter.hooks.builtin.budget",
    "RateLimitHook": "agent_adapter.hooks.builtin.rate_limit",
    "Ledger": "agent_adapter.cost.ledger",
    "LedgerEntry": "agent_adapter.core.types",
    "Prices": "agent_adapter.cost.prices",
    "MemoryCache": "agent_adapter.cache.memory",
    # Flows
    "Flow": "agent_adapter.flow.spec",
    "FlowRunner": "agent_adapter.flow.runner",
    # Model messages
    "LLMRequest": "agent_adapter.core.messages",
    "LLMResponse": "agent_adapter.core.messages",
    "Message": "agent_adapter.core.messages",
    "ToolDef": "agent_adapter.core.messages",
    "Event": "agent_adapter.core.types",
    # Model ports that need no extra
    "CallableModel": "agent_adapter.models.callable",
    "FallbackModel": "agent_adapter.models.fallback",
    # Errors (the five most caught; the rest are in agent_adapter.errors)
    "AdapterError": "agent_adapter.core.errors",
    "ToolError": "agent_adapter.core.errors",
    "ApprovalDenied": "agent_adapter.core.errors",
    "ApprovalPending": "agent_adapter.core.errors",
    "MissingDependency": "agent_adapter.core.errors",
    # The default registry, created on first access
    "default_registry": "agent_adapter.core._default",
    "tool": "agent_adapter.core._default",
    "wrap": "agent_adapter.core._default",
}

# Names that left the top level in 1.0 -> their public home.
_MOVED: dict[str, str] = {
    **dict.fromkeys(
        (
            "ApprovalAlreadyDecided",
            "ApprovalNotFound",
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
            "ToolFailed",
            "ToolNotFound",
            "ToolTimeout",
            "ToolUsageError",
            "Unauthorized",
        ),
        "agent_adapter.errors",
    ),
    "ShortCircuit": "agent_adapter.hooks",
}

# Subpackages reachable as attributes after a bare `import agent_adapter`.
_SUBPACKAGES = frozenset(
    {
        "approval",
        "audit",
        "cache",
        "chat",
        "cli",
        "context",
        "core",
        "cost",
        "embeddings",
        "evals",
        "handlers",
        "errors",
        "exec",
        "flow",
        "hooks",
        "integrations",
        "memory",
        "models",
        "optimiser",
        "policy",
        "retrieval",
        "runtime",
        "sandbox",
        "schema",
        "server",
        "tracing",
        "viewer",
    }
)


def _getattr(name: str) -> "Any":
    """Load a public name or a subpackage on first access; explain names that moved."""
    import importlib

    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
    elif name in _SUBPACKAGES:
        value = importlib.import_module(f"{__name__}.{name}")
    elif name in _MOVED:
        message = f"{name} moved: from {_MOVED[name]} import {name}"
        if _in_from_import():  # `from x import y` would replace an AttributeError's text
            raise ImportError(message, name=__name__)
        raise AttributeError(message, name=name)
    else:
        hint = _hint(name)
        if hint and _in_from_import():
            raise ImportError(f"cannot import name {name!r} from {__name__!r}; {hint}", name=name)
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; "
            + (hint or "see agent_adapter.__all__ for the top-level names"),
            name=name,
        )
    globals()[name] = value  # later lookups skip __getattr__
    return value


def _hint(name: str) -> str:
    """Where a missing name lives, or the close top-level name; "" when nothing fits."""
    if name.startswith("_"):
        return ""  # dunder probes (copy, pickle, IDEs) stay cheap
    homes = _homes().get(name)
    if homes:
        return f"it lives in {', '.join(homes)}: from {homes[0]} import {name}"
    import difflib

    close = difflib.get_close_matches(name, [*__all__, *_MOVED], n=1, cutoff=0.8)
    if close:
        return f"did you mean {close[0]!r}? see agent_adapter.__all__ for the top-level names"
    return ""


_HOMES: "dict[str, list[str]] | None" = None


def _homes() -> "dict[str, list[str]]":
    """Name -> public modules that export it, read from the package source (no imports)."""
    global _HOMES
    if _HOMES is None:
        from pathlib import Path

        root = Path(__file__).parent
        files = [*root.glob("*/__init__.py"), *root.glob("integrations/*/__init__.py")]
        files += [p for p in root.glob("integrations/*.py") if not p.name.startswith("_")]
        homes: dict[str, list[str]] = {}
        for path in sorted(files):
            parts = path.relative_to(root.parent).with_suffix("").parts
            module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            for exported in _exports(path):
                if exported not in _LAZY:
                    homes.setdefault(exported, []).append(module)
        _HOMES = homes
    return _HOMES


def _exports(path: "Path") -> "tuple[str, ...]":
    """The literal `__all__` of a source file; () when unreadable or not a literal."""
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            try:
                names = ast.literal_eval(node.value)
            except ValueError:
                return ()
            return tuple(n for n in names if isinstance(n, str))
    return ()


def _in_from_import() -> bool:
    """True when the code that triggered `__getattr__` runs `from agent_adapter import ...`."""
    import opcode
    import sys

    try:
        frame = sys._getframe(3)  # 0: here, 1: _getattr, 2: __getattr__, 3: the importer
        return frame.f_code.co_code[frame.f_lasti] == opcode.opmap["IMPORT_FROM"]
    except (AttributeError, IndexError, KeyError, ValueError):
        return False  # no CPython frames: the AttributeError message still applies


if not TYPE_CHECKING:  # hidden from type checkers, so an unknown name stays a type error

    def __getattr__(name: str) -> "Any":
        """PEP 562 hook: see `_getattr`."""
        return _getattr(name)


def __dir__() -> list[str]:
    """The public names plus any subpackage already loaded."""
    return sorted({*__all__, *(name for name in globals() if name in _SUBPACKAGES)})


__all__ = [
    "AdapterError",
    "ApprovalDecision",
    "ApprovalDenied",
    "ApprovalPending",
    "ApprovalRequest",
    "AutoApprove",
    "AutoDeny",
    "BudgetHook",
    "CallableModel",
    "CallbackApproval",
    "Decision",
    "DeferredApproval",
    "Event",
    "Executor",
    "FallbackModel",
    "Flow",
    "FlowRunner",
    "GuardrailHook",
    "Hook",
    "LLMRequest",
    "LLMResponse",
    "Ledger",
    "LedgerEntry",
    "LoggingHook",
    "MemoryApprovalStore",
    "MemoryCache",
    "Message",
    "MissingDependency",
    "Prices",
    "RateLimitHook",
    "RedisApprovalStore",
    "Registry",
    "RunLimits",
    "SQLiteApprovalStore",
    "Step",
    "ToolCall",
    "ToolContext",
    "ToolDef",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "__version__",
    "acall",
    "attach",
    "build_spec",
    "call",
    "configure",
    "default_registry",
    "run_sync",
    "run_tool_loop",
    "set_default_executor",
    "spec_from_schema",
    "tool",
    "wrap",
]
