"""A host's tunable hooks: which hooks are on the surface, their configs and validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import OptimiserError
from agent_adapter.hooks.base import Hook, hook_name
from agent_adapter.hooks.builtin import BudgetHook, GuardrailHook, LoggingHook, RateLimitHook
from agent_adapter.optimiser._transcript import TrackedHook
from agent_adapter.optimiser.model import HarnessTree
from agent_adapter.optimiser.surface import ParsedPath, parse_path


def parsed_or_none(path: str) -> ParsedPath | None:
    """The parsed path, or None for a path `validate_tree` already reports."""
    try:
        return parse_path(path)
    except OptimiserError:
        return None


def _config_text(hook: Hook) -> str:
    return json.dumps(hook.config(), indent=2, sort_keys=True, ensure_ascii=False)


# Hooks that enforce limits or protect data: tuning them could loosen a guardrail.
SAFETY_HOOKS: tuple[type[Hook], ...] = (BudgetHook, GuardrailHook, RateLimitHook, LoggingHook)


def is_tunable(hook: Hook, *, allow_safety_tuning: bool = False) -> bool:
    """A hook with a `config_schema()`; safety hooks only when the host allows it."""
    if type(hook).config_schema() is None:
        return False
    return allow_safety_tuning or not isinstance(hook, SAFETY_HOOKS)


class HookSurface:
    """The tunable hooks of a host, keyed by name (safety hooks run but are never edited)."""

    def __init__(self, hooks: Sequence[Hook], *, allow_safety_tuning: bool) -> None:
        self.hooks = tuple(hooks)
        tunable = [h for h in self.hooks if is_tunable(h, allow_safety_tuning=allow_safety_tuning)]
        names = [hook_name(h) for h in tunable]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise OptimiserError(f"duplicate hook names among tunable hooks: {duplicates}")
        self.tunable = dict(zip(names, tunable, strict=True))

    def files(self) -> dict[str, str]:
        return {f"hook/{name}/config.json": _config_text(h) for name, h in self.tunable.items()}

    def schemas(self) -> dict[str, dict[str, Any]]:
        return {name: type(h).config_schema() or {} for name, h in self.tunable.items()}

    def problems(self, tree: HarnessTree) -> list[str]:
        problems: list[str] = []
        for path, text in tree.files.items():
            parsed = parsed_or_none(path)
            if parsed is None or parsed.kind != "hook":
                continue
            name = parsed.name
            hook = self.tunable.get(name)
            if hook is None:
                problems.append(f"{path}: no tunable hook named {name!r}")
                continue
            try:
                hook.with_config(json.loads(text))
            except Exception as exc:  # the hook's own validation (ConfigError, ValueError, ...)
                if is_control_flow(exc):
                    raise
                problems.append(f"{path}: {exc}")
        return problems

    def applied(self, tree: HarnessTree) -> list[TrackedHook]:
        """Every hook, tunable ones with the tree's config (unchanged configs keep the instance)."""
        out: list[TrackedHook] = []
        for hook in self.hooks:
            name = hook_name(hook)
            tunable = self.tunable.get(name) is hook
            text = tree.files.get(f"hook/{name}/config.json") if tunable else None
            if text is not None:
                config = json.loads(text)
                if config != hook.config():
                    hook = hook.with_config(config)
            out.append(TrackedHook(hook))
        return out
