import logging

import pytest

from agent_adapter.core.errors import ConfigError, GuardrailBlocked, ToolUsageError
from agent_adapter.core.types import ErrorInfo, ToolResult
from agent_adapter.hooks.builtin.guardrail import GuardrailHook
from tests.hooks.builtin.helpers import llm_inv, llm_response, ok, tool_inv

ALL = ("args", "result", "llm_request", "llm_response")


async def run_all_stages(hook: GuardrailHook) -> None:
    inv = tool_inv()
    await hook.before_tool(inv)
    await hook.after_tool(inv, ok("ana"))
    linv = llm_inv()
    await hook.before_llm(linv)
    await hook.after_llm(linv, llm_response())


async def test_default_checks_only_args():
    seen = []
    hook = GuardrailHook(lambda stage, payload: seen.append((stage, payload)) is None)
    await run_all_stages(hook)
    assert seen == [("args", {"user": "ana", "password": "hunter2"})]


async def test_each_stage_gets_its_payload_and_passing_returns_none():
    seen = []

    async def check(stage, payload):
        seen.append((stage, payload))
        return True

    hook = GuardrailHook(check, on=ALL)
    inv, linv, response = tool_inv(), llm_inv(), llm_response()
    assert await hook.before_tool(inv) is None
    assert await hook.after_tool(inv, ok("ana")) is None
    assert await hook.before_llm(linv) is None
    assert await hook.after_llm(linv, response) is None
    assert seen == [
        ("args", inv.call.arguments),
        ("result", "ana"),
        ("llm_request", linv.request),
        ("llm_response", response),
    ]


async def test_failed_results_are_not_checked():
    seen = []
    hook = GuardrailHook(lambda stage, payload: seen.append(stage) is None, on=["result"])
    failed = ToolResult("c", "login", "error", error=ErrorInfo(kind="tool-error", message="x"))
    assert await hook.after_tool(tool_inv(), failed) is None
    assert await hook.before_tool(tool_inv()) is None  # "args" is not a chosen stage
    assert seen == []


@pytest.mark.parametrize(
    ("verdict", "reason"),
    [(False, "check failed"), ("contains a password", "contains a password"), ("", "check failed")],
)
async def test_false_or_a_reason_blocks(verdict, reason, caplog):
    hook = GuardrailHook(lambda stage, payload: verdict)
    with caplog.at_level(logging.WARNING), pytest.raises(GuardrailBlocked) as info:
        await hook.before_tool(tool_inv())
    error = info.value
    assert (error.reason, error.stage, error.tool) == (reason, "args", "login")
    assert "guardrail blocked args for tool 'login'" in caplog.text
    assert "hunter2" not in caplog.text
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == [
        "guardrail blocked args for tool 'login'"
    ]


async def test_the_reason_is_logged_at_debug_only(caplog):
    hook = GuardrailHook(lambda stage, payload: "amount 9000 over limit")
    with caplog.at_level(logging.DEBUG), pytest.raises(GuardrailBlocked):
        await hook.before_tool(tool_inv())
    [debug] = [r for r in caplog.records if "amount 9000" in r.getMessage()]
    assert debug.levelno == logging.DEBUG


@pytest.mark.parametrize("stage", ["llm_request", "llm_response"])
async def test_llm_stages_block_with_no_tool(stage):
    hook = GuardrailHook(lambda s, payload: "off-topic", on=[stage])
    linv = llm_inv()
    with pytest.raises(GuardrailBlocked) as info:
        if stage == "llm_request":
            await hook.before_llm(linv)
        else:
            await hook.after_llm(linv, llm_response())
    assert (info.value.stage, info.value.tool) == (stage, "")


async def test_result_stage_blocks():
    hook = GuardrailHook(lambda stage, payload: payload != "ana", on="result")
    with pytest.raises(GuardrailBlocked) as info:
        await hook.after_tool(tool_inv(), ok("ana"))
    assert info.value.stage == "result"


async def test_a_non_bool_non_str_verdict_is_a_usage_error():
    with pytest.raises(ToolUsageError, match="bool or a reason string"):
        await GuardrailHook(lambda stage, payload: None).before_tool(tool_inv())


async def test_check_errors_propagate():
    def check(stage, payload):
        raise RuntimeError("classifier down")

    with pytest.raises(RuntimeError, match="classifier down"):
        await GuardrailHook(check).before_tool(tool_inv())


@pytest.mark.parametrize("on", [[], ["args", "output"], "everything"])
def test_bad_stages_are_rejected(on):
    with pytest.raises(ConfigError, match="stage"):
        GuardrailHook(lambda stage, payload: True, on=on)


def test_check_must_be_callable():
    with pytest.raises(ConfigError, match="callable"):
        GuardrailHook("not a function")  # type: ignore[arg-type]


def test_stages_are_exposed():
    assert GuardrailHook(lambda s, p: True, on=["result", "args"]).stages == frozenset(
        {"args", "result"}
    )
