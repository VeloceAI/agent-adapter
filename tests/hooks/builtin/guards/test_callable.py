import threading

import pytest

from agent_adapter.core.errors import ConfigError, GuardrailBlocked
from agent_adapter.hooks.builtin.guards import CallableGuard, GuardVerdict, as_verdict
from tests.hooks.builtin.helpers import llm_inv, llm_response, ok, tool_inv


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (True, GuardVerdict(True)),
        (False, GuardVerdict(False, "check failed")),
        ("off topic", GuardVerdict(False, "off topic")),
        ("", GuardVerdict(False, "check failed")),
        (GuardVerdict(False, "x", 1), GuardVerdict(False, "x", 1)),
    ],
)
def test_as_verdict(answer, expected):
    assert as_verdict(answer) == expected


def test_as_verdict_rejects_other_types():
    with pytest.raises(TypeError):
        as_verdict(1)


async def test_async_fn_sees_stage_and_payload():
    seen = []

    async def service(stage, payload):
        seen.append((stage, payload))
        return True

    guard = CallableGuard(service, on=("args", "result", "llm_request", "llm_response"))
    inv, linv, response = tool_inv(), llm_inv(), llm_response()
    await guard.before_tool(inv)
    await guard.after_tool(inv, ok("v"))
    await guard.before_llm(linv)
    await guard.after_llm(linv, response)
    assert seen == [
        ("args", inv.call.arguments),
        ("result", "v"),
        ("llm_request", linv.request),
        ("llm_response", response),
    ]


async def test_sync_fn_runs_off_the_event_loop():
    threads = []

    def service(stage, payload):
        threads.append(threading.current_thread())
        return "blocked by policy"

    with pytest.raises(GuardrailBlocked, match="blocked by policy"):
        await CallableGuard(service, on="args").before_tool(tool_inv())
    assert threads and threads[0] is not threading.main_thread()


async def test_sync_fn_returning_an_awaitable_is_awaited():
    async def later():
        return False

    with pytest.raises(GuardrailBlocked, match="check failed"):
        await CallableGuard(lambda s, p: later(), on="args").before_tool(tool_inv())


async def test_redact_uses_the_replacement():
    guard = CallableGuard(
        lambda s, p: GuardVerdict(False, "masked", "***"), on="result", action="redact"
    )
    assert (await guard.after_tool(tool_inv(), ok("secret"))).value == "***"


async def test_an_external_error_blocks():
    def down(stage, payload):
        raise ConnectionError("service down")

    with pytest.raises(GuardrailBlocked, match="guard failed"):
        await CallableGuard(down, on="args").before_tool(tool_inv())


async def test_a_bad_answer_blocks():
    with pytest.raises(GuardrailBlocked, match="guard failed"):
        await CallableGuard(lambda s, p: 3, on="args").before_tool(tool_inv())


def test_fn_must_be_callable():
    with pytest.raises(ConfigError, match="callable"):
        CallableGuard("nope")
