import json
import logging

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolError
from agent_adapter.core.types import ErrorInfo, ToolCall, ToolResult
from agent_adapter.hooks import ToolInvocation
from agent_adapter.hooks.builtin.logging import DEFAULT_REDACT, REDACTED, LoggingHook, redact
from tests.hooks.builtin.helpers import llm_inv, llm_response, node_inv, ok, tool_inv

LOGGER = "test.logging_hook"


@pytest.mark.parametrize(
    ("key", "hidden"),
    [
        ("api_key", True),
        ("API_KEY", True),
        ("x-api-key", True),
        ("access_token", True),
        ("client_secret", True),
        ("password", True),
        ("Authorization", True),
        ("Set-Cookie", True),
        ("user", False),
        ("query", False),
        ("passcode", False),
    ],
)
def test_redaction_table(key, hidden):
    assert (redact({key: "v"})[key] == REDACTED) is hidden


def test_redaction_walks_nested_mappings_and_lists_without_mutating():
    payload = {
        "user": "ana",
        "headers": {"Authorization": "Bearer abc", "accept": "json"},
        "items": [{"token": "t1", "id": 1}, "plain"],
        "pair": ({"secret": 1},),
    }
    out = redact(payload)
    assert out == {
        "user": "ana",
        "headers": {"Authorization": REDACTED, "accept": "json"},
        "items": [{"token": REDACTED, "id": 1}, "plain"],
        "pair": [{"secret": REDACTED}],
    }
    assert payload["headers"]["Authorization"] == "Bearer abc"


def test_redaction_custom_patterns():
    assert redact({"ssn": 1, "token": 2}, ("SSN",)) == {"ssn": REDACTED, "token": 2}
    assert DEFAULT_REDACT == ("key", "token", "secret", "password", "authorization", "cookie")


def lines(caplog) -> list[tuple[int, str, dict]]:
    return [(r.levelno, r.aa_event, r.aa_fields) for r in caplog.records if hasattr(r, "aa_event")]


@pytest.fixture
def hook():
    return LoggingHook(logger=logging.getLogger(LOGGER))


async def test_tool_start_finish_and_error_are_one_line_each(hook, caplog):
    inv = tool_inv(arguments={"user": "ana", "password": "hunter2", "settings": {"api_key": "k"}})
    error = ToolResult(
        "call_1", "login", "error", error=ErrorInfo(kind="tool-error", message="bad"), attempts=2
    )
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert await hook.before_tool(inv) is None
        assert await hook.after_tool(inv, ok()) is None
        assert await hook.after_tool(inv, error) is None
        assert await hook.on_tool_error(inv, ToolError("nope", tool="login")) is None
    (l1, e1, f1), (l2, e2, f2), (_, e3, f3), (l4, e4, f4) = lines(caplog)
    assert (l1, e1) == (logging.INFO, "tool.start")
    assert f1 == {
        "tool": "login",
        "kind": "tool",
        "run_id": "run_1",
        "call_id": "call_1",
        "args": {"user": "ana", "password": REDACTED, "settings": {"api_key": REDACTED}},
    }
    assert (l2, e2) == (logging.INFO, "tool.finish")
    assert f2["status"] == "ok" and f2["attempts"] == 1 and f2["latency_ms"] == 3.0
    assert "value" not in f2 and "error_kind" not in f2
    assert (e3, f3["status"], f3["error_kind"]) == ("tool.finish", "error", "tool-error")
    assert (l4, e4) == (logging.WARNING, "tool.error")
    assert f4["error_kind"] == "tool-error"
    assert f4["error_type"] == "agent_adapter.core.errors.ToolError"
    assert "hunter2" not in caplog.text and "'k'" not in caplog.text
    for fields in (f1, f2, f3, f4):  # every line can be tied to its run and call
        assert (fields["run_id"], fields["call_id"]) == ("run_1", "call_1")


async def test_tool_call_id_falls_back_to_the_context(hook, caplog):
    inv = tool_inv(ToolContext(run_id="run_9", call_id="call_ctx"))
    inv = ToolInvocation(inv.spec, ToolCall(name="login", arguments=inv.call.arguments), inv.ctx)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await hook.before_tool(inv)
        await hook.on_tool_error(inv, ValueError())
    assert [(f["run_id"], f["call_id"]) for _, _, f in lines(caplog)] == [
        ("run_9", "call_ctx"),
        ("run_9", "call_ctx"),
    ]


async def test_message_is_a_json_line(hook, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await hook.before_tool(tool_inv())
    event, body = caplog.records[0].getMessage().split(" ", 1)
    assert event == "tool.start" and json.loads(body)["tool"] == "login"


@pytest.mark.parametrize("confirm", [True, lambda call, ctx: True])
async def test_confirm_tool_arguments_are_never_logged(hook, caplog, confirm):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await hook.before_tool(tool_inv(confirm=confirm, arguments={"user": "ana-secret-name"}))
    (_, _, fields) = lines(caplog)[0]
    assert "args" not in fields and fields["args_withheld"] is True
    assert "ana-secret-name" not in caplog.text


async def test_llm_lines_never_carry_message_content(hook, caplog):
    inv = llm_inv()
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await hook.before_llm(inv)
        await hook.after_llm(inv, llm_response())
        await hook.on_llm_error(inv, TimeoutError())
    (_, e1, f1), (_, e2, f2), (l3, e3, f3) = lines(caplog)
    assert (e1, f1) == (
        "llm.start",
        {
            "provider": "fake",
            "model": "m1",
            "stage": "plan",
            "run_id": "run_1",
            "messages": 1,
            "tools": 0,
            "stream": True,
        },
    )
    assert e2 == "llm.finish"
    assert f2["input_tokens"] == 10 and f2["output_tokens"] == 5 and f2["finish_reason"] == "stop"
    assert f2["cached_input_tokens"] == 2 and f2["tool_calls"] == 0 and f2["cached"] is False
    assert (l3, e3, f3["error_kind"], f3["error_type"]) == (
        logging.WARNING,
        "llm.error",
        "TimeoutError",
        "builtins.TimeoutError",
    )
    assert "secret" not in caplog.text
    assert f2["run_id"] == f3["run_id"] == "run_1"


async def test_node_lines(hook, caplog):
    inv = node_inv()
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await hook.before_node(inv)
        await hook.after_node(inv, {"label": "x", "score": 1})
        await hook.after_node(inv, None)
        await hook.on_node_error(inv, ValueError("bad"))
    events = lines(caplog)
    assert [e for _, e, _ in events] == ["node.start", "node.finish", "node.finish", "node.error"]
    assert events[0][2] == {"flow": "triage", "node": "classify", "run_id": "run_1"}
    assert events[1][2]["updated"] == ["label", "score"] and events[2][2]["updated"] == []
    assert events[3][2]["error_kind"] == "ValueError"
    assert {fields["run_id"] for _, _, fields in events} == {"run_1"}


async def test_level_is_configurable_and_disabled_levels_skip_work(caplog):
    quiet = LoggingHook(logger=logging.getLogger(LOGGER), level=logging.DEBUG)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await quiet.before_tool(tool_inv())
        await quiet.after_tool(tool_inv(), ok())
        await quiet.on_tool_error(tool_inv(), ValueError())  # errors are at least WARNING
    assert [e for _, e, _ in lines(caplog)] == ["tool.error"]


def test_default_logger_is_the_module_logger():
    assert LoggingHook().logger.name == "agent_adapter.hooks.builtin.logging"


async def test_tool_start_masks_every_secret_name_whatever_the_patterns(caplog):
    custom = LoggingHook(logger=logging.getLogger(LOGGER), redact=("ssn",))
    inv = tool_inv(
        arguments={"ssn": "1", "passwd": "p", "api_keys": ["k1"], "key": "plainsecret", "q": "x"}
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        await custom.before_tool(inv)
    ((_, _, fields),) = lines(caplog)
    assert fields["args"] == {
        "ssn": REDACTED,
        "passwd": REDACTED,
        "api_keys": REDACTED,
        "key": REDACTED,
        "q": "x",
    }
