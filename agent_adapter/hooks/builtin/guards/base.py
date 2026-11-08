"""ContentGuard: a hook that asks a guard about each stage's payload and blocks, redacts or asks."""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, NoReturn, get_args

from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import ConfigError, GuardrailBlocked, qualified_name
from agent_adapter.core.messages import LLMResponse
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ToolCall, ToolResult
from agent_adapter.hooks.base import Hook, LLMInvocation, ToolInvocation
from agent_adapter.hooks.builtin.guardrail import Stage
from agent_adapter.hooks.builtin.guards._walk import Path, fields, rewrite
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)

_KIND = re.compile(r"[a-z][a-z0-9_.-]{0,39}")  # a label such as `email`, never a found value
_MAX_KINDS = 20

Action = Literal["block", "redact", "approve"]
FIXED_STAGES = frozenset({"args", "llm_request"})  # hooks cannot rewrite these payloads
DEFAULT_STAGES: tuple[Stage, ...] = ("args", "result", "llm_response")
GUARD_FAILED = "guard failed"
NEEDS_CONFIRM = "needs approval: set confirm=guard.confirm on the tool"


class _Unset:
    def __repr__(self) -> str:
        return "NO_REPLACEMENT"


NO_REPLACEMENT: Any = _Unset()


@dataclass(frozen=True)
class GuardVerdict:
    """A guard's answer: allowed, or a reason and (to redact) the rewritten payload.

    `kinds` labels what was found (`email`, `card`) and `count` how many: they go into the
    `guardrail.redacted` event, so they must never hold the found values.
    """

    allowed: bool
    reason: str = ""
    replacement: Any = NO_REPLACEMENT
    kinds: tuple[str, ...] = ()
    count: int = 0


class ContentGuard(Hook):
    """Base guard: `inspect(stage, payload)` decides; `action` blocks, redacts or asks."""

    name = "guard"

    def __init__(
        self, *, on: Iterable[Stage] | Stage = DEFAULT_STAGES, action: Action = "block"
    ) -> None:
        stages = frozenset([on] if isinstance(on, str) else on)
        allowed = frozenset(get_args(Stage))
        if not stages or not stages <= allowed:
            raise ConfigError(
                f"{type(self).__name__} stages must be a non-empty subset of {sorted(allowed)}, "
                f"got {sorted(stages)}"
            )
        if action not in get_args(Action):
            raise ConfigError(f"guard action must be one of {list(get_args(Action))}")
        if action == "redact" and stages & FIXED_STAGES:
            raise ConfigError(
                "redact rewrites results and model replies only; use block or approve "
                f"for {sorted(stages & FIXED_STAGES)}"
            )
        if action == "approve" and stages != {"args"}:
            raise ConfigError("approve applies to tool arguments only: use on=('args',)")
        self.stages: frozenset[str] = stages
        self.action: Action = action

    async def inspect(self, stage: str, payload: Any) -> GuardVerdict:
        """The verdict on one payload; subclasses implement it."""
        raise NotImplementedError

    async def confirm(self, call: ToolCall, ctx: ToolContext) -> bool:
        """A `ToolSpec.confirm` rule: True (ask for approval) when the arguments fail the guard."""
        verdict = await self._verdict("args", call.arguments, call.name)
        return not verdict.allowed

    async def before_tool(self, inv: ToolInvocation) -> None:
        if "args" not in self.stages:
            return
        tool = inv.spec.name
        verdict = await self._verdict("args", inv.call.arguments, tool)
        if verdict.allowed:
            return
        if self.action == "approve":
            if self._approval_follows(inv.spec):
                logger.info("guard %s sent tool %s to approval", self.name, tool)
                return
            self._block("args", tool, NEEDS_CONFIRM)
        self._block("args", tool, verdict.reason)

    async def after_tool(self, inv: ToolInvocation, result: ToolResult) -> ToolResult | None:
        if "result" not in self.stages or not result.ok:
            return None
        tool = inv.spec.name
        verdict = await self._verdict("result", result.value, tool)
        if verdict.allowed:
            return None
        value = self._replacement("result", tool, verdict)
        logger.info("guard %s redacted the result of tool %s", self.name, tool)
        self._redacted(inv.ctx, "result", verdict, tool=tool)
        return dataclasses.replace(result, value=value)

    async def before_llm(self, inv: LLMInvocation) -> None:
        if "llm_request" in self.stages:
            verdict = await self._verdict("llm_request", inv.request, "")
            if not verdict.allowed:
                self._block("llm_request", "", verdict.reason)

    async def after_llm(self, inv: LLMInvocation, response: LLMResponse) -> LLMResponse | None:
        if "llm_response" not in self.stages:
            return None
        verdict = await self._verdict("llm_response", response, "")
        if verdict.allowed:
            return None
        replaced = self._replacement("llm_response", "", verdict)
        if not isinstance(replaced, LLMResponse):
            self._block("llm_response", "", verdict.reason)
        logger.info("guard %s redacted a %s reply", self.name, inv.provider)
        self._redacted(inv.ctx, "llm_response", verdict, provider=inv.provider, model=inv.model)
        return replaced

    async def _verdict(self, stage: str, payload: Any, tool: str) -> GuardVerdict:
        try:
            verdict = await self.inspect(stage, payload)
        except Exception as exc:
            if is_control_flow(exc) or isinstance(exc, GuardrailBlocked):
                raise
            # Fail closed; only the type is logged: guard errors can quote the payload.
            logger.error(
                "guard %s failed on %s%s (%s); blocked",
                self.name,
                stage,
                _where(tool),
                qualified_name(type(exc)),
            )
            raise GuardrailBlocked(GUARD_FAILED, tool=tool, stage=stage) from exc
        if not isinstance(verdict, GuardVerdict):
            logger.error("guard %s returned %s, not a GuardVerdict; blocked", self.name, stage)
            raise GuardrailBlocked(GUARD_FAILED, tool=tool, stage=stage)
        return verdict

    def _replacement(self, stage: str, tool: str, verdict: GuardVerdict) -> Any:
        if self.action != "redact" or verdict.replacement is NO_REPLACEMENT:
            self._block(stage, tool, verdict.reason)
        return verdict.replacement

    def _redacted(self, ctx: ToolContext, stage: str, verdict: GuardVerdict, **where: str) -> None:
        kinds = sorted({k for k in verdict.kinds if isinstance(k, str) and _KIND.fullmatch(k)})
        count = verdict.count if isinstance(verdict.count, int) and verdict.count > 0 else 1
        event = make_event(
            EventType.GUARDRAIL_REDACTED,
            ctx,
            guard=self.name,
            stage=stage,
            count=count,
            kinds=kinds[:_MAX_KINDS],
            **where,
        )
        emit(ctx.sink, event)

    def _approval_follows(self, spec: ToolSpec) -> bool:
        rule = spec.confirm
        return rule is True or (callable(rule) and rule == self.confirm)

    def _block(self, stage: str, tool: str, reason: str) -> NoReturn:
        logger.warning("guard %s blocked %s%s", self.name, stage, _where(tool))
        raise GuardrailBlocked(reason or "blocked by guard", tool=tool, stage=stage)


class TextGuard(ContentGuard):
    """A guard that reviews each text in a payload; redaction swaps each text for its rewrite."""

    async def review(self, text: str, stage: str) -> GuardVerdict:
        """The verdict on one text of `stage`; `replacement` is its redacted text."""
        raise NotImplementedError

    def review_field(self, keys: tuple[str, ...], text: str) -> GuardVerdict | None:
        """A verdict from the keys above an allowed text (`password`, `auth`), or None."""
        return None

    async def inspect(self, stage: str, payload: Any) -> GuardVerdict:
        verdicts: dict[str, GuardVerdict] = {}  # each distinct text is reviewed once
        failed: dict[Path, GuardVerdict] = {}  # by place: equal texts elsewhere stay as they are
        for field in fields(stage, payload):
            verdict = verdicts.get(field.text)
            if verdict is None:
                verdict = await self.review(field.text, stage)
                if not isinstance(verdict, GuardVerdict):
                    raise TypeError(
                        f"review must return a GuardVerdict, got {type(verdict).__name__}"
                    )
                verdicts[field.text] = verdict
            if verdict.allowed and field.keys:
                verdict = self.review_field(field.keys, field.text) or verdict
            if not verdict.allowed:
                failed[field.path] = verdict
        if not failed:
            return GuardVerdict(True)
        reason = "; ".join(dict.fromkeys(v.reason for v in failed.values() if v.reason))
        if self.action != "redact" or stage in FIXED_STAGES:
            return GuardVerdict(False, reason)
        swaps = {path: v.replacement for path, v in failed.items()}
        if not all(isinstance(new, str) for new in swaps.values()):
            return GuardVerdict(False, reason)  # a text without a rewrite cannot be redacted
        kinds = tuple(sorted({kind for v in failed.values() for kind in v.kinds}))
        count = sum(max(v.count, 1) for v in failed.values())
        changed = rewrite(stage, payload, lambda p, s: swaps.get(p, s))
        return GuardVerdict(False, reason, changed, kinds=kinds, count=count)


def _where(tool: str) -> str:
    return f" for tool {tool!r}" if tool else ""
