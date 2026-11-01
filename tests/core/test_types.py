import asyncio
import json
import typing
from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel, ValidationError

from agent_adapter.core.errors import (
    ApprovalPending,
    ToolArgsInvalid,
    ToolError,
    ToolFailed,
    ToolTimeout,
)
from agent_adapter.core.types import (
    ApprovalDecision,
    ApprovalRequest,
    Decision,
    ErrorInfo,
    Event,
    LedgerEntry,
    ModelCaps,
    ToolCall,
    ToolResult,
    Usage,
)


def test_tool_call_defaults_and_frozen():
    call = ToolCall(name="search")
    assert (call.arguments, call.id) == ({}, "")
    with pytest.raises(ValidationError):
        call.name = "x"


def test_error_info_from_adapter_error_keeps_message_kind_and_retry_flag():
    info = ErrorInfo.from_exception(ToolTimeout("t", 2.0))
    assert info.kind == "tool-timeout"
    assert info.message == "tool 't' timed out after 2s"
    assert info.exception == "agent_adapter.core.errors.ToolTimeout"
    assert info.retryable is True
    assert info.details == {}


def test_error_info_from_foreign_exception_hides_the_message():
    info = ErrorInfo.from_exception(ValueError("secret=abc"))
    assert info.kind == "internal-error"
    assert info.message == "ValueError"
    assert info.exception == "ValueError"
    assert info.retryable is True


def test_error_info_never_marks_base_exceptions_retryable():
    assert ErrorInfo.from_exception(asyncio.CancelledError()).retryable is False


def test_error_info_details_per_kind():
    args = ToolArgsInvalid("t", [{"type": "missing", "loc": ("a",), "msg": "required"}])
    assert ErrorInfo.from_exception(args).details == {
        "errors": [{"type": "missing", "loc": ["a"], "msg": "required"}]
    }
    assert ErrorInfo.from_exception(ApprovalPending("apr_1")).details == {"approval_id": "apr_1"}
    try:
        raise ToolFailed("t") from KeyError("k")
    except ToolFailed as exc:
        assert ErrorInfo.from_exception(exc).details == {"cause_type": "KeyError"}
    tool_error = ToolError("no rows", details={"rows": 0})
    info = ErrorInfo.from_exception(tool_error)
    assert info.details == {"rows": 0} and info.message == "no rows"


def test_error_info_is_json_safe():
    info = ErrorInfo.from_exception(ToolArgsInvalid("t", [{"type": "x", "loc": (), "msg": "m"}]))
    assert json.loads(info.model_dump_json())["kind"] == "tool-args-invalid"


def test_tool_result_ok_and_unwrap():
    result = ToolResult(call_id="c", name="t", status="ok", value=3)
    assert result.ok and result.unwrap() == 3
    assert (result.attempts, result.cached, result.latency_ms) == (0, False, 0.0)
    with pytest.raises(FrozenInstanceError):
        result.value = 4  # type: ignore[misc]


def test_tool_result_unwrap_reraises_the_exception():
    error = ToolTimeout("t", 1.0)
    result = ToolResult("c", "t", "error", error=ErrorInfo.from_exception(error), exception=error)
    assert not result.ok
    with pytest.raises(ToolTimeout) as raised:
        result.unwrap()
    assert raised.value is error


def test_tool_result_unwrap_without_exception_raises_tool_failed():
    info = ErrorInfo(kind="tool-error", message="remote said no")
    result = ToolResult("c1", "t", "error", error=info, attempts=2)
    with pytest.raises(ToolFailed) as raised:
        result.unwrap()
    assert (raised.value.tool, raised.value.call_id, raised.value.attempts) == ("t", "c1", 2)


def test_tool_result_exception_is_not_compared_or_shown():
    a = ToolResult("c", "t", "error", exception=ValueError("x"))
    b = ToolResult("c", "t", "error", exception=KeyError("y"))
    assert a == b
    assert "exception" not in repr(a)


class Hit(BaseModel):
    title: str


class Opaque:
    pass


def test_tool_result_to_dict_is_json_safe():
    result = ToolResult(
        "c",
        "t",
        "ok",
        value={"hits": [Hit(title="a")], "seen": {1}, "obj": Opaque()},
        latency_ms=1.5,
        cached=True,
    )
    data = result.to_dict()
    assert json.loads(json.dumps(data))["value"]["hits"] == [{"title": "a"}]
    assert data["value"]["seen"] == [1]
    assert isinstance(data["value"]["obj"], str)
    assert set(data) == {
        "call_id",
        "name",
        "status",
        "value",
        "error",
        "attempts",
        "cached",
        "latency_ms",
    }
    assert data["error"] is None


def test_tool_result_to_dict_dumps_error():
    info = ErrorInfo(kind="tool-error", message="m")
    data = ToolResult("c", "t", "error", error=info).to_dict()
    assert data["error"] == {
        "kind": "tool-error",
        "message": "m",
        "exception": "",
        "retryable": False,
        "details": {},
    }


def test_event_defaults_and_to_dict():
    event = Event("tool.started", run_id="r", data={"tool": "t", "when": {2}})
    assert event.ts > 0 and event.call_id == "" and event.parent_id == ""
    data = event.to_dict()
    assert data == {
        "type": "tool.started",
        "run_id": "r",
        "call_id": "",
        "parent_id": "",
        "ts": event.ts,
        "data": {"tool": "t", "when": [2]},
    }


def test_usage_adds_and_counts_prompt_tokens():
    a = Usage(input_tokens=10, cached_input_tokens=5, cache_write_by_ttl={"5m": 3}, units=1.5)
    b = Usage(
        input_tokens=1,
        cache_write_input_tokens=2,
        cache_write_by_ttl={"5m": 1, "1h": 2},
        output_tokens=7,
        reasoning_tokens=4,
        units=0.5,
    )
    total = a + b
    assert total == Usage(
        input_tokens=11,
        cached_input_tokens=5,
        cache_write_input_tokens=2,
        cache_write_by_ttl={"5m": 4, "1h": 2},
        output_tokens=7,
        reasoning_tokens=4,
        units=2.0,
    )
    assert total.prompt_tokens == 18
    assert a.__add__(3) is NotImplemented


def test_usage_rejects_negative_counts():
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1)
    with pytest.raises(ValidationError):
        Usage(cache_write_by_ttl={"5m": -1})


def test_ledger_entry_defaults_entry_id_to_call_id():
    entry = LedgerEntry(run_id="r", call_id="c", kind="llm")
    assert entry.entry_id == "c"
    assert LedgerEntry(run_id="r", call_id="c", entry_id="e", kind="tool").entry_id == "e"
    assert entry.cost_usd is None and entry.ok and entry.attempts == 1


def test_ledger_entry_from_usage_copies_counts():
    usage = Usage(
        input_tokens=3,
        cached_input_tokens=2,
        cache_write_input_tokens=4,
        cache_write_by_ttl={"1h": 4},
        output_tokens=5,
        reasoning_tokens=1,
        units=1.0,
    )
    entry = LedgerEntry.from_usage(usage, run_id="r", call_id="c", kind="llm", model="m")
    assert (entry.input_tokens, entry.cached_input_tokens, entry.output_tokens) == (3, 2, 5)
    assert (entry.cache_write_input_tokens, entry.reasoning_tokens) == (4, 1)
    assert entry.cache_write_by_ttl == {"1h": 4}
    assert LedgerEntry(run_id="r", call_id="c", kind="api").cache_write_by_ttl == {}
    assert entry.units == 1.0 and entry.model == "m" and entry.entry_id == "c"


def test_approval_types():
    request = ApprovalRequest(approval_id="apr_1", run_id="r", call_id="c", tool="t")
    assert request.kind == "tool" and request.arguments == {} and request.created_at == 0.0
    decision = ApprovalDecision(approved=True, edited_args={"x": 1})
    assert Decision is ApprovalDecision
    assert decision.reason == "" and decision.decided_by == ""
    with pytest.raises(ValidationError):
        ApprovalRequest(approval_id="a", run_id="r", call_id="c", tool="t", kind="other")


def test_model_caps_defaults_disable_gated_features():
    caps = ModelCaps()
    assert not (caps.prompt_cache_key or caps.explicit_cache or caps.cache_control)
    assert caps.min_cache_tokens == 1024 and caps.structured_output and not caps.reasoning


class Unserialisable:
    def __str__(self) -> str:
        return "<unserialisable>"


def test_error_info_details_are_json_safe():
    info = ErrorInfo.from_exception(ToolError("bad", details={"obj": Unserialisable(), "n": {1}}))
    assert info.details == {"obj": "<unserialisable>", "n": [1]}
    data = ToolResult("c", "t", "error", error=info).to_dict()
    assert json.loads(json.dumps(data))["error"]["details"]["obj"] == "<unserialisable>"


def test_foreign_exception_with_a_kind_attribute_does_not_leak_its_message():
    class Sneaky(Exception):
        kind = "tool-error"
        retryable = False
        details: typing.ClassVar = {"password": "hunter2"}

    info = ErrorInfo.from_exception(Sneaky("password=hunter2"))
    assert info.kind == "internal-error" and info.message == "Sneaky"
    assert info.details == {} and info.retryable is True
    assert info.exception.endswith("Sneaky")


def test_tool_result_unwrap_pending_raises_approval_pending():
    info = ErrorInfo.from_exception(ApprovalPending("apr_9"))
    result = ToolResult("c1", "t", "pending", error=info)
    with pytest.raises(ApprovalPending) as raised:
        result.unwrap()
    assert (raised.value.approval_id, raised.value.tool, raised.value.call_id) == (
        "apr_9",
        "t",
        "c1",
    )
    with pytest.raises(ApprovalPending) as bare:
        ToolResult("c1", "t", "pending").unwrap()
    assert bare.value.approval_id == ""


def test_cache_scope_has_one_definition():
    import typing

    from agent_adapter.cache import keys
    from agent_adapter.core import _policy, tool, types

    assert typing.get_args(types.CacheScope) == ("run", "thread", "global")
    assert keys.CacheScope is types.CacheScope and tool.CacheScope is types.CacheScope
    assert _policy.CACHE_SCOPES == ("run", "thread", "global")


def test_non_utf8_bytes_in_details_are_base64():
    info = ErrorInfo.from_exception(ToolError("bad", details={"raw": b"\xff", "ok": b"ok"}))
    assert info.details == {"raw": "_w==", "ok": "b2s="}
    data = ToolResult("c", "t", "ok", value=b"\xff\xfe").to_dict()
    assert json.loads(json.dumps(data))["value"] == "__4="


def test_qualified_name_has_one_implementation():
    import inspect

    from agent_adapter.core import types

    assert "__qualname__" not in inspect.getsource(types)
    assert ErrorInfo.from_exception(ValueError()).exception == "ValueError"
    assert ErrorInfo.from_exception(ToolError("x")).exception == (
        "agent_adapter.core.errors.ToolError"
    )
