import asyncio
import logging

import pytest

from agent_adapter import ApprovalDecision, Registry
from agent_adapter.approval.builtin import AutoApprove, CallbackApproval
from agent_adapter.core.errors import GuardrailBlocked
from agent_adapter.core.types import ErrorInfo, ToolCall, ToolResult
from agent_adapter.exec import Executor
from agent_adapter.hooks.builtin.guards import (
    NO_REPLACEMENT,
    CallableGuard,
    ContentGuard,
    GuardVerdict,
    PIIGuard,
    TextGuard,
)
from tests.hooks.builtin.helpers import llm_inv, llm_response, ok, tool_inv


def shop(guard, **policy):
    registry = Registry()

    @registry.tool(**policy)
    def find_customer(name: str) -> dict:
        """Look a customer up."""
        return {"name": name, "email": "ana@example.com"}

    return registry, find_customer


async def test_executor_blocks_args_before_the_tool_runs():
    registry, _ = shop(None)
    executor = Executor(registry, hooks=[PIIGuard(on="args")])
    result = await executor.run_call(
        ToolCall(name="find_customer", arguments={"name": "ana@example.com"})
    )
    assert result.status == "error"
    assert result.error.kind == "guardrail-blocked"


async def test_executor_redacts_the_result_the_model_sees():
    registry, _ = shop(None)
    executor = Executor(registry, hooks=[PIIGuard(on="result", action="redact")])
    value = await executor.run("find_customer", {"name": "Ana"})
    assert value == {"name": "Ana", "email": "[EMAIL]"}


async def test_approve_asks_the_port_only_when_the_guard_flags():
    guard = PIIGuard(on="args", action="approve")
    registry, _ = shop(guard, confirm=guard.confirm)
    asked = []

    def decide(request):
        asked.append(request.tool)
        return ApprovalDecision(approved=True, decided_by="reviewer")

    executor = Executor(registry, hooks=[guard], approval=CallbackApproval(decide))
    assert (await executor.run("find_customer", {"name": "Ana"}))["name"] == "Ana"
    assert asked == []
    await executor.run("find_customer", {"name": "ana@example.com"})
    assert asked == ["find_customer"]


async def test_approve_without_a_port_is_denied():
    guard = PIIGuard(on="args", action="approve")
    registry, _ = shop(guard, confirm=guard.confirm)
    result = await Executor(registry, hooks=[guard]).run_call(
        ToolCall(name="find_customer", arguments={"name": "ana@example.com"})
    )
    assert result.error.kind == "approval-denied"


async def test_approve_on_a_tool_without_the_rule_blocks():
    guard = PIIGuard(on="args", action="approve")
    with pytest.raises(GuardrailBlocked, match=r"confirm=guard.confirm"):
        await guard.before_tool(tool_inv(arguments={"user": "ana@example.com", "password": "x"}))


async def test_approve_passes_when_the_tool_always_confirms():
    guard = PIIGuard(on="args", action="approve")
    registry, _ = shop(guard, confirm=True)
    executor = Executor(registry, hooks=[guard], approval=AutoApprove())
    assert (await executor.run("find_customer", {"name": "ana@example.com"}))["name"]


class Broken(ContentGuard):
    def __init__(self, error, **kw):
        super().__init__(**kw)
        self.error = error

    async def inspect(self, stage, payload):
        raise self.error


async def test_a_guard_that_raises_blocks_and_logs_only_the_type(caplog):
    guard = Broken(RuntimeError("leaked ana@example.com"), on=("args", "result"))
    with caplog.at_level(logging.DEBUG), pytest.raises(GuardrailBlocked) as info:
        await guard.before_tool(tool_inv())
    assert info.value.reason == "guard failed"
    assert "RuntimeError" in caplog.text
    assert "leaked" not in caplog.text


async def test_control_flow_passes_through():
    with pytest.raises(asyncio.CancelledError):
        await Broken(asyncio.CancelledError()).before_tool(tool_inv())


async def test_a_non_verdict_blocks():
    guard = Broken(None)
    guard.inspect = lambda stage, payload: asyncio.sleep(0, "yes")
    with pytest.raises(GuardrailBlocked, match="guard failed"):
        await guard.before_tool(tool_inv())


async def test_base_inspect_is_abstract():
    with pytest.raises(GuardrailBlocked):
        await ContentGuard().before_tool(tool_inv())
    with pytest.raises(GuardrailBlocked):
        await TextGuard().before_tool(tool_inv())


async def test_failed_results_and_unchosen_stages_are_skipped():
    guard = PIIGuard(on="llm_response", action="redact")
    failed = ToolResult("c", "login", "error", error=ErrorInfo(kind="tool-error", message="x"))
    assert await guard.after_tool(tool_inv(), failed) is None
    assert await guard.before_tool(tool_inv(arguments={"user": "a@example.com"})) is None
    assert await guard.before_llm(llm_inv()) is None
    assert await PIIGuard(on="args").after_llm(llm_inv(), llm_response("a@b.example")) is None
    assert await PIIGuard(on="args").after_tool(tool_inv(), ok("a@b.example")) is None
    assert await PIIGuard(on="result").before_tool(tool_inv(arguments={"u": "a@b.example"})) is None


async def test_redact_without_a_rewrite_blocks():
    guard = CallableGuard(
        lambda stage, payload: GuardVerdict(False, "no"), on="result", action="redact"
    )
    with pytest.raises(GuardrailBlocked, match="no"):
        await guard.after_tool(tool_inv(), ok("x"))


async def test_a_reply_rewrite_that_is_not_a_response_blocks():
    guard = CallableGuard(
        lambda stage, payload: GuardVerdict(False, "no", "plain text"),
        on="llm_response",
        action="redact",
    )
    with pytest.raises(GuardrailBlocked):
        await guard.after_llm(llm_inv(), llm_response())


class Partial(TextGuard):
    async def review(self, text, stage):
        if "a" in text:
            return GuardVerdict(False, "has a", text.replace("a", "*"))
        if "b" in text:
            return GuardVerdict(False, "has b")  # no rewrite
        return GuardVerdict(True)


async def test_text_guard_blocks_when_any_text_has_no_rewrite():
    guard = Partial(on="result", action="redact")
    assert (await guard.after_tool(tool_inv(), ok(["xa", "c"]))).value == ["x*", "c"]
    with pytest.raises(GuardrailBlocked, match="has a; has b"):
        await guard.after_tool(tool_inv(), ok(["xa", "b"]))


async def test_text_guard_rejects_a_bad_review():
    class Bad(TextGuard):
        async def review(self, text, stage):
            return "nope"

    with pytest.raises(GuardrailBlocked, match="guard failed"):
        await Bad(on="result").after_tool(tool_inv(), ok("x"))


def test_no_replacement_repr():
    assert repr(NO_REPLACEMENT) == "NO_REPLACEMENT"


async def test_block_without_reason_has_a_default():
    guard = CallableGuard(lambda s, p: GuardVerdict(False), on="args")
    with pytest.raises(GuardrailBlocked, match="blocked by guard"):
        await guard.before_tool(tool_inv())


async def test_a_guard_may_raise_its_own_block():
    with pytest.raises(GuardrailBlocked, match="mine"):
        await Broken(GuardrailBlocked("mine")).before_tool(tool_inv())
