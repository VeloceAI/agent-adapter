import asyncio
import logging

import pytest
from pydantic import BaseModel

from agent_adapter.core import control
from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import register_control_flow
from agent_adapter.core.errors import BudgetExceeded, GuardrailBlocked, ToolUsageError
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.tool import build_spec
from agent_adapter.core.types import ToolCall, ToolResult
from agent_adapter.hooks import (
    Hook,
    LLMInvocation,
    NodeInvocation,
    ShortCircuit,
    ToolInvocation,
    hook_name,
    run_after,
    run_before,
    run_on_error,
)
from tests.fakes.hooks import RaisingHook, RecordingHook


def ping(x: int) -> int:
    """Echo."""
    return x


TOOL = ToolInvocation(build_spec(ping), ToolCall(name="ping", arguments={"x": 1}), ToolContext())
LLM = LLMInvocation(LLMRequest(messages=()), "fake", "m", ToolContext())


class State(BaseModel):
    n: int = 0


NODE = NodeInvocation("f", "n", State(), ToolContext())


def result(value) -> ToolResult:
    return ToolResult("c", "ping", "ok", value=value)


async def test_base_hook_does_nothing():
    hook = Hook()
    assert await hook.before_tool(TOOL) is None
    assert await hook.after_tool(TOOL, result(1)) is None
    assert await hook.on_tool_error(TOOL, ValueError()) is None
    assert await hook.before_llm(LLM) is None
    response = LLMResponse(provider="fake", model="m", message=Message(role="assistant"))
    assert await hook.after_llm(LLM, response) is None
    assert await hook.on_llm_error(LLM, ValueError()) is None
    assert await hook.before_node(NODE) is None
    assert await hook.after_node(NODE, {"n": 1}) is None
    assert await hook.on_node_error(NODE, ValueError()) is None
    assert hook.config() == {} and Hook.config_schema() is None


def test_with_config_is_unsupported_by_default():
    with pytest.raises(ToolUsageError, match="'Hook' does not support with_config"):
        Hook().with_config({})


def test_hook_name_defaults_to_the_class_name():
    class Named(Hook):
        name = "custom"

    assert hook_name(Hook()) == "Hook" and hook_name(Named()) == "custom"


async def test_before_runs_in_order_and_the_first_short_circuit_wins():
    first, third = RecordingHook(), RecordingHook()
    winner = RecordingHook({"before_tool": ShortCircuit("cached")})
    outcome = await run_before([first, winner, third], "before_tool", TOOL)
    assert outcome == ShortCircuit("cached")
    assert first.phases() == ["before_tool"] and third.calls == []
    assert await run_before([first], "before_llm", LLM) is None


async def test_after_threads_the_value_and_none_keeps_it():
    double = RecordingHook({"after_tool": result(2)})
    keep = RecordingHook()
    value = await run_after([keep, double, keep], "after_tool", TOOL, result(1))
    assert value == result(2)
    assert keep.calls[1][1] == (TOOL, result(2))
    assert await run_after([], "after_node", NODE, None) is None


async def test_on_error_first_short_circuit_recovers():
    error = ValueError("x")
    skip, fix = RecordingHook(), RecordingHook({"on_node_error": ShortCircuit({"n": 1})})
    assert await run_on_error([skip, fix], "on_node_error", NODE, error) == ShortCircuit({"n": 1})
    assert skip.calls == [("on_node_error", (NODE, error))]
    assert await run_on_error([skip], "on_llm_error", LLM, error) is None


async def test_a_raising_hook_is_logged_and_reraised_unchanged(caplog):
    error = RuntimeError("hook bug")
    after = RecordingHook()
    with (
        caplog.at_level(logging.ERROR, logger="agent_adapter.hooks.dispatch"),
        pytest.raises(RuntimeError) as raised,
    ):
        await run_before([RaisingHook(error), after], "before_tool", TOOL)
    assert raised.value is error and after.calls == []
    assert "hook raising failed in before_tool" in caplog.text


@pytest.mark.parametrize(
    "error", [BudgetExceeded(1.0, 1.5), GuardrailBlocked("pii in args", tool="t")]
)
async def test_policy_outcomes_are_debug_only_since_the_deciding_hook_logs_them(caplog, error):
    with (
        caplog.at_level(logging.DEBUG, logger="agent_adapter.hooks.dispatch"),
        pytest.raises(type(error)) as raised,
    ):
        await run_before([RaisingHook(error)], "before_tool", TOOL)
    assert raised.value is error
    [record] = caplog.records
    assert record.levelno == logging.DEBUG and record.exc_info is None
    assert (
        f"hook raising stopped before_tool: agent_adapter.core.errors.{type(error).__name__}"
        in record.getMessage()
    )
    assert "pii in args" not in record.getMessage()


async def test_registered_control_flow_from_a_hook_is_not_logged(caplog, monkeypatch):
    class HostSignal(Exception):
        pass

    monkeypatch.setattr(control, "_registered", ())
    register_control_flow(HostSignal)
    with (
        caplog.at_level(logging.DEBUG, logger="agent_adapter.hooks.dispatch"),
        pytest.raises(HostSignal),
    ):
        await run_before([RaisingHook(HostSignal())], "before_tool", TOOL)
    assert caplog.records == []


async def test_control_flow_from_a_hook_is_not_logged(caplog):
    class GraphBubbleUp(Exception):
        pass

    class GraphInterrupt(GraphBubbleUp):
        pass

    with caplog.at_level(logging.ERROR, logger="agent_adapter.hooks.dispatch"):
        with pytest.raises(GraphInterrupt):
            await run_after([RaisingHook(GraphInterrupt())], "after_tool", TOOL, result(1))
        with pytest.raises(asyncio.CancelledError):
            await run_on_error([RaisingHook(asyncio.CancelledError())], "on_tool_error", TOOL, None)
    assert caplog.text == ""


async def test_wrong_return_types_and_phases_are_usage_errors():
    bad = RecordingHook({"before_tool": "not a short circuit"})
    with pytest.raises(ToolUsageError, match="must return ShortCircuit or None, got str"):
        await run_before([bad], "before_tool", TOOL)
    bad_error = RecordingHook({"on_tool_error": 3})
    with pytest.raises(ToolUsageError, match="got int"):
        await run_on_error([bad_error], "on_tool_error", TOOL, ValueError())
    with pytest.raises(ToolUsageError, match="'after_tool' is not one of"):
        await run_before([], "after_tool", TOOL)
    with pytest.raises(ToolUsageError):
        await run_after([], "before_tool", TOOL, None)
    with pytest.raises(ToolUsageError):
        await run_on_error([], "on_flow_error", TOOL, ValueError())
