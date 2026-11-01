"""Validate the execution-policy options accepted by build_spec."""

from __future__ import annotations

import math
import typing
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from agent_adapter.core.errors import ToolDefinitionError
from agent_adapter.core.types import CacheScope

Check = Callable[[Any], tuple[bool, Any]]


def _non_empty_str(value: Any) -> tuple[bool, Any]:
    return isinstance(value, str) and bool(value), value


def _tags(value: Any) -> tuple[bool, Any]:
    if isinstance(value, str):
        return bool(value), frozenset({value})
    if not isinstance(value, Iterable):
        return False, value
    tags = list(value)
    return all(isinstance(tag, str) and tag for tag in tags), frozenset(tags)


def _targets(value: Any) -> tuple[bool, Any]:
    if value is None:
        return True, None
    ok, normalised = _tags(value)
    return ok and bool(normalised), normalised


def _bool(value: Any) -> tuple[bool, Any]:
    return isinstance(value, bool), value


def _confirm(value: Any) -> tuple[bool, Any]:
    return isinstance(value, bool) or callable(value), value


def _positive_or_none(value: Any) -> tuple[bool, Any]:
    if value is None:
        return True, value
    number = isinstance(value, int | float) and not isinstance(value, bool)
    return number and math.isfinite(value) and value > 0, value


def _retries(value: Any) -> tuple[bool, Any]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0, value


def _str_or_none(value: Any) -> tuple[bool, Any]:
    return value is None or isinstance(value, str), value


def _bool_or_none(value: Any) -> tuple[bool, Any]:
    return value is None or isinstance(value, bool), value


def _one_of(*choices: str | None) -> Check:
    def check(value: Any) -> tuple[bool, Any]:
        return isinstance(value, str | None) and value in choices, value

    return check


CACHE_SCOPES: tuple[str, ...] = typing.get_args(CacheScope)
EFFECTS = ("read_only", "additive", "destructive")


_CHECKS: dict[str, tuple[Check, str]] = {
    "kind": (_non_empty_str, "a non-empty str"),
    "tags": (_tags, "a non-empty str or an iterable of non-empty str"),
    "parallel_safe": (_bool, "a bool"),
    "once_per_run": (_bool, "a bool"),
    "portable": (_bool, "a bool"),
    "confirm": (_confirm, "a bool or a callable"),
    "timeout": (_positive_or_none, "a finite positive number or None"),
    "cache_ttl": (_positive_or_none, "a finite positive number or None"),
    "retries": (_retries, "an int >= 0"),
    "prompt_hint": (_str_or_none, "a str or None"),
    "cache_scope": (_one_of(*CACHE_SCOPES), "one of 'run', 'thread', 'global'"),
    "targets": (_targets, "None, a non-empty str or a non-empty iterable of non-empty str"),
    "effects": (_one_of(*EFFECTS, None), "one of 'read_only', 'additive', 'destructive' or None"),
    "idempotent": (_bool_or_none, "a bool or None"),
}
POLICY_OPTIONS = frozenset(_CHECKS)


def validate_policy(tool_name: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Return the policy options with normalised values.

    Raises:
        ToolDefinitionError: an option is unknown or has a value of the wrong type.
    """
    unknown = set(policy) - POLICY_OPTIONS
    if unknown:
        raise ToolDefinitionError(f"tool {tool_name!r}: unknown option(s) {sorted(unknown)}")
    clean: dict[str, Any] = {}
    for option, value in policy.items():
        check, expected = _CHECKS[option]
        ok, normalised = check(value)
        if not ok:
            raise ToolDefinitionError(
                f"tool {tool_name!r}: option {option!r} must be {expected}, got {value!r}"
            )
        clean[option] = normalised
    return clean
