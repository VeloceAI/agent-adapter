"""Find integrations through the `agent_adapter.integrations` entry-point group."""

from __future__ import annotations

import importlib.metadata
import logging
import threading
from typing import Any

from agent_adapter.core.errors import IntegrationError, log_failure
from agent_adapter.integrations.base import Integration, _check_requires

logger = logging.getLogger(__name__)

GROUP = "agent_adapter.integrations"

_lock = threading.Lock()  # guards `_found` and `_generation`; held only briefly
_discovery_lock = threading.Lock()  # one discovery at a time; plugins load under it
_found: dict[str, Integration] | None = None
_generation = 0  # bumped by every completed discovery
_local = threading.local()  # `discovering` is True while this thread loads plugins


def available_integrations(*, refresh: bool = False) -> dict[str, Integration]:
    """Every integration that loads and conforms, by name; cached until `refresh=True`.

    Concurrent calls share one discovery. Raises IntegrationError when called by a plugin
    while it is being discovered.
    """
    return _available(refresh, caller="available_integrations")


def get_integration(name: str) -> Integration:
    """The integration registered as `name`, with its required modules importable.

    Raises:
        IntegrationError: no integration has that name, or called during discovery.
        MissingDependency: a module it requires is not installed.
    """
    found = _available(False, caller="get_integration")
    integration = found.get(name)
    if integration is None:
        raise IntegrationError(
            f"no integration named {name!r}; available: {sorted(found)}", integration=name
        )
    _check_requires(integration)
    return integration


def _available(refresh: bool, *, caller: str) -> dict[str, Integration]:
    global _found, _generation
    # Checked before any lock: a plugin calling back from its own import would deadlock.
    if getattr(_local, "discovering", False):
        raise IntegrationError(
            f"{caller} called during discovery: a plugin may not look up "
            "integrations while it is imported or created"
        )
    with _lock:
        if not refresh and _found is not None:
            return dict(_found)
        seen = _generation
    with _discovery_lock:
        with _lock:
            if _generation != seen and _found is not None:  # another thread just discovered
                return dict(_found)
        _local.discovering = True
        try:
            found = _discover()
        finally:
            _local.discovering = False
        with _lock:
            _found, _generation = found, _generation + 1
    return dict(found)


def _discover() -> dict[str, Integration]:
    found: dict[str, Integration] = {}
    for point in importlib.metadata.entry_points(group=GROUP):
        integration = _load(point)
        if integration is None:
            continue
        if point.name in found:
            logger.warning(
                "integration %r is registered more than once; keeping the first, skipping %s",
                point.name,
                _origin(point),
            )
            continue
        found[point.name] = integration
    logger.debug("integrations available: %s", sorted(found))
    return found


def _load(point: Any) -> Integration | None:
    """The entry point's integration, or None (logged) when it cannot be used."""
    try:
        obj = point.load()
    except ModuleNotFoundError as exc:
        if not _is_own_module(point, exc.name):
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "integration %r failed to load from %s; skipped",
                point.name,
                _origin(point),
            )
            return None
        # Expected for a built-in whose module is not released yet, or an uninstalled plugin.
        logger.warning(
            "integration %r skipped: %s is not installed (%s)", point.name, exc.name, _origin(point)
        )
        return None
    except Exception as exc:
        log_failure(
            logger,
            logging.WARNING,
            exc,
            "integration %r failed to load from %s; skipped",
            point.name,
            _origin(point),
        )
        return None
    if isinstance(obj, type):
        try:
            obj = obj()
        except Exception as exc:
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "integration %r could not be created from %s; skipped",
                point.name,
                _origin(point),
            )
            return None
    problem = _problem(point.name, obj)
    if problem:
        logger.warning("integration %r from %s skipped: %s", point.name, _origin(point), problem)
        return None
    integration: Integration = obj
    return integration


def _problem(name: str, obj: Any) -> str:
    """Why `obj` is not a usable integration registered as `name`; "" when it is."""
    if not isinstance(obj, Integration):
        return "it does not implement the Integration protocol"
    if not isinstance(obj.name, str):
        return "name must be a str"
    if not (isinstance(obj.requires, tuple) and all(isinstance(r, str) for r in obj.requires)):
        return "requires must be a tuple of str"
    if not isinstance(obj.extra, str):
        return "extra must be a str"
    if obj.name != name:
        return f"its name {obj.name!r} does not match the entry-point name {name!r}"
    return ""


def _is_own_module(point: Any, missing: str | None) -> bool:
    """True when `missing` is the entry point's module or one of its parent packages."""
    module = str(point.value).partition(":")[0].strip()
    return bool(missing) and (module == missing or module.startswith(f"{missing}."))


def _origin(point: Any) -> str:
    dist = getattr(point, "dist", None)
    return f"{point.value} ({dist.name})" if dist is not None else str(point.value)


__all__ = ["GROUP", "available_integrations", "get_integration"]
