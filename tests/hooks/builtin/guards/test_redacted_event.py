"""A redacting guard sends `guardrail.redacted`: counts and kind labels, never the values."""

from __future__ import annotations

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.hooks.builtin.guards import GuardVerdict, PIIGuard
from agent_adapter.hooks.builtin.guards.base import ContentGuard
from agent_adapter.tracing.sinks import EventLog
from tests.hooks.builtin.helpers import llm_inv, llm_response, ok, tool_inv

EMAIL, CARD = "ann@example.com", "4111 1111 1111 1111"


def context() -> tuple[ToolContext, EventLog]:
    log = EventLog()
    return ToolContext(run_id="run_1", call_id="call_1", sink=log), log


async def test_a_redacted_result_reports_kinds_and_count() -> None:
    ctx, log = context()
    guard = PIIGuard(on="result", action="redact")
    await guard.after_tool(tool_inv(ctx), ok({"note": f"{EMAIL} pays {CARD}", "cc": EMAIL}))
    (event,) = log.of_type("guardrail.redacted")
    assert dict(event.data) == {
        "guard": "pii_guard",
        "stage": "result",
        "count": 3,
        "kinds": ["card", "email"],
        "tool": "login",
    }
    assert EMAIL not in repr(event) and "4111" not in repr(event)


async def test_a_redacted_reply_names_the_model() -> None:
    ctx, log = context()
    guard = PIIGuard(on="llm_response", action="redact")
    await guard.after_llm(llm_inv(ctx), llm_response(f"write to {EMAIL}"))
    (event,) = log.of_type("guardrail.redacted")
    assert dict(event.data) == {
        "guard": "pii_guard",
        "stage": "llm_response",
        "count": 2,  # the reply text and its message
        "kinds": ["email"],
        "provider": "fake",
        "model": "m1",
    }


async def test_blocking_and_allowing_send_no_redaction() -> None:
    ctx, log = context()
    await PIIGuard(on="result", action="redact").after_tool(tool_inv(ctx), ok("clean"))
    with pytest.raises(Exception, match="found"):
        await PIIGuard(on="result").after_tool(tool_inv(ctx), ok(EMAIL))
    assert log.of_type("guardrail.redacted") == []


class Rewriter(ContentGuard):
    """A custom guard whose labels are not all safe."""

    def __init__(self, kinds: tuple[str, ...], count: int) -> None:
        super().__init__(on="result", action="redact")
        self._verdict_kinds, self._count = kinds, count

    async def inspect(self, stage: str, payload: object) -> GuardVerdict:
        return GuardVerdict(False, "x", "[gone]", kinds=self._verdict_kinds, count=self._count)


async def test_custom_kinds_that_look_like_values_are_dropped() -> None:
    ctx, log = context()
    guard = Rewriter(("email", EMAIL, "Has Space", "x" * 50), 0)
    await guard.after_tool(tool_inv(ctx), ok("anything"))
    (event,) = log.of_type("guardrail.redacted")
    assert event.data["kinds"] == ["email"] and event.data["count"] == 1


async def test_no_sink_means_no_event_and_no_error() -> None:
    guard = PIIGuard(on="result", action="redact")
    result = await guard.after_tool(tool_inv(), ok(EMAIL))
    assert result is not None and result.value == "[EMAIL]"
