"""NemoGuard: NVIDIA NeMo Guardrails rails check each payload (`LLMRails.check_async`)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent_adapter.core.errors import ConfigError, MissingDependency
from agent_adapter.hooks.builtin.guardrail import Stage
from agent_adapter.hooks.builtin.guards.base import DEFAULT_STAGES, Action, GuardVerdict, TextGuard

INSTALL = 'pip install "agent-adapter[nemo]"'
# Arguments and requests go in as the user's turn (input rails); results and replies as the
# assistant's turn (output rails).
INPUT_STAGES = frozenset({"args", "llm_request"})


class NemoGuard(TextGuard):
    """NeMo Guardrails as a guard: pass `rails` (an `LLMRails`) or `config` (its folder)."""

    name = "nemo_guard"

    def __init__(
        self,
        rails: Any = None,
        *,
        config: str | Path | None = None,
        on: Iterable[Stage] | Stage = DEFAULT_STAGES,
        action: Action = "block",
    ) -> None:
        super().__init__(on=on, action=action)
        if (rails is None) == (config is None):
            raise ConfigError("NemoGuard takes exactly one of rails= or config=")
        self.rails = rails if rails is not None else _load(Path(config or ""))

    async def review(self, text: str, stage: str) -> GuardVerdict:
        return await self.review_as(text, "user" if stage in INPUT_STAGES else "assistant")

    async def review_as(self, text: str, role: str) -> GuardVerdict:
        """The rails' verdict on `text` sent as a `role` turn."""
        result = await self.rails.check_async([{"role": role, "content": text}])
        status = str(getattr(result.status, "value", result.status)).upper()
        if status == "PASSED":
            return GuardVerdict(True)
        if status == "MODIFIED":
            return GuardVerdict(False, "modified by rails", str(result.content))
        if status == "BLOCKED":
            rail = getattr(result, "rail", None)
            return GuardVerdict(False, f"blocked by rail {rail}" if rail else "blocked by rails")
        raise ValueError(f"unknown rails status {status!r}")


def _load(config: Path) -> Any:
    try:
        from nemoguardrails import LLMRails, RailsConfig
    except ImportError as exc:
        raise MissingDependency(f"NemoGuard needs nemoguardrails: {INSTALL}") from exc
    return LLMRails(RailsConfig.from_path(str(config)))
