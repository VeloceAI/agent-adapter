"""`attach`: a framework's native tools, found by name or from one of its objects."""

from __future__ import annotations

import difflib
import logging
import types
from collections.abc import Mapping
from typing import Any

from agent_adapter.core.errors import IntegrationError, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.integrations.base import (
    Integration,
    ToolRunner,
    _check_requires,
    _importable,
)
from agent_adapter.integrations.discovery import available_integrations

logger = logging.getLogger(__name__)

# Option names that would smuggle a framework approval past the executor.
_APPROVAL_WORDS = ("approv", "confirm", "consent")
_APPROVAL_HINT = (
    "approval is never passed to attach: set it on the Executor "
    "(Executor(registry, approval=DeferredApproval(store))) and pass executor="
)


def attach(
    target_or_name: Any,
    registry: Registry | None = None,
    *,
    executor: ToolRunner | None = None,
    **opts: Any,
) -> Any:
    """The framework's native tools for `registry` (default: `default_registry`).

    `target_or_name` is an integration name (`"langgraph"`, `"openai-agents"`) or an
    object, class or module from that framework. `executor` runs every call (default: the
    process-wide default executor). Approval stays on the executor: an option such as
    `approval=True` is refused. Other `opts` go to an integration's own `attach` method.

    Raises:
        ToolUsageError: an approval option, a bad registry or executor, or options the
            integration does not take.
        IntegrationError: no installed integration matches; the message lists them.
        MissingDependency: the integration's extra is not installed.
    """
    _refuse_approval(opts)
    registry = _registry(target_or_name, registry)
    _check_executor(executor)
    found = available_integrations()
    if isinstance(target_or_name, str):
        integration = _by_name(target_or_name, found)
    else:
        integration = _by_object(target_or_name, found)
    _check_requires(integration)
    hook = getattr(integration, "attach", None)
    native = None if isinstance(target_or_name, str) else target_or_name
    if callable(hook):
        logger.debug("attaching through %r's own attach", integration.name)
        return hook(native, registry, executor=executor, **opts)
    if opts:
        raise ToolUsageError(
            f"integration {integration.name!r} takes no attach options {sorted(opts)}; "
            f"use agent_adapter.integrations.{integration.name} directly for those"
        )
    count = len(registry.for_target(integration.name))
    if not count:
        logger.warning("no tools in the registry are exported to %r", integration.name)
    logger.debug("attaching %d tool(s) to %r", count, integration.name)
    export = getattr(integration, "export_tools", None)
    if callable(export):
        return export(registry, executor=executor)
    return [
        integration.export_tool(spec, executor=executor)
        for spec in registry.for_target(integration.name)
    ]


def _refuse_approval(opts: Mapping[str, Any]) -> None:
    """Raise ToolUsageError for any option that names an approval."""
    named = sorted(key for key in opts if any(word in key.casefold() for word in _APPROVAL_WORDS))
    if named:
        raise ToolUsageError(f"attach does not take {', '.join(named)}: {_APPROVAL_HINT}")


def _registry(target: Any, registry: Any) -> Registry:
    """`registry`, or the default registry when None."""
    if isinstance(target, Registry):
        raise ToolUsageError("pass the framework first: attach('<framework>', registry)")
    if registry is None:
        from agent_adapter.core._default import default_registry

        return default_registry
    if not isinstance(registry, Registry):
        raise ToolUsageError(f"registry must be a Registry or None, got {registry!r}")
    return registry


def _check_executor(executor: Any) -> None:
    """Raise ToolUsageError unless `executor` is None or has a `run_spec` method."""
    if executor is not None and not callable(getattr(executor, "run_spec", None)):
        raise ToolUsageError(
            f"executor must be an Executor (or another ToolRunner), got {executor!r}; "
            f"{_APPROVAL_HINT}"
        )


def _key(name: str) -> str:
    return name.strip().casefold().replace("-", "_").replace(" ", "_").replace(".", "_")


def _by_name(name: str, found: Mapping[str, Integration]) -> Integration:
    """The integration whose name or extra matches `name`, ignoring case, `-` and `_`."""
    wanted = _key(name)
    for integration in found.values():
        if wanted in (_key(integration.name), _key(integration.extra)):
            return integration
    close = difflib.get_close_matches(wanted, [_key(n) for n in found], n=1)
    hint = f"; did you mean {close[0]!r}?" if close else ""
    raise IntegrationError(
        f"no integration named {name!r}{hint} {catalogue(found)}", integration=name
    )


def _by_object(target: Any, found: Mapping[str, Integration]) -> Integration:
    """The integration whose import names own `target`'s class (or module)."""
    modules = _modules(target)
    scores: dict[str, tuple[int, int, int]] = {}
    for name, integration in found.items():
        roots = integration.requires or (integration.name,)
        hits = [
            (depth, rank, -len(root))
            for depth, module in enumerate(modules)
            for rank, root in enumerate(roots)
            if module == root or module.startswith(f"{root}.")
        ]
        if hits:
            scores[name] = min(hits)
    what = f"{type(target).__name__} from {modules[0] if modules else 'builtins'}"
    if not scores:
        raise IntegrationError(
            f"cannot tell which framework a {what} belongs to; pass a name instead, "
            f"e.g. attach('langgraph') {catalogue(found)}"
        )
    best = min(scores.values())
    winners = sorted(name for name, score in scores.items() if score == best)
    if len(winners) > 1:
        raise IntegrationError(f"a {what} matches {winners}; pass one of those names to attach")
    return found[winners[0]]


def _modules(target: Any) -> list[str]:
    """Module names that define `target`, its class and its base classes, nearest first."""
    if isinstance(target, types.ModuleType):
        return [target.__name__]
    cls = target if isinstance(target, type) else type(target)
    names = [str(getattr(base, "__module__", "")) for base in cls.__mro__]
    return [name for name in dict.fromkeys(names) if name and name != "builtins"]


def catalogue(found: Mapping[str, Integration]) -> str:
    """`(installed: ...; needs an extra: name (pip install ...), ...)` for error messages."""
    ready = sorted(n for n, i in found.items() if all(_importable(r) for r in i.requires))
    missing = sorted(set(found) - set(ready))
    extras = [
        f'{n} (pip install "agent-adapter[{found[n].extra}]")' if found[n].extra else n
        for n in missing
    ]
    return (
        f"(installed: {', '.join(ready) or 'none'}; needs an extra: {', '.join(extras) or 'none'})"
    )
