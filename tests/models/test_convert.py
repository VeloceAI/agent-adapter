import logging
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agent_adapter.core.errors import ModelOutputInvalid
from agent_adapter.core.messages import LLMResponse, Message, OutputSchema
from agent_adapter.core.types import ToolCall
from agent_adapter.models import _convert
from agent_adapter.models._convert import (
    count,
    dump,
    dumps_arguments,
    get,
    items,
    loads_arguments,
    tool_names,
    tool_payload,
    warn_unsupported,
    with_parsed,
)

SCHEMA = OutputSchema(name="Answer", json_schema={"type": "object"})


def _reply(text="", finish="stop", calls=()):
    return LLMResponse(
        provider="p",
        model="m",
        id="r1",
        text=text,
        tool_calls=calls,
        finish_reason=finish,
        message=Message(role="assistant", content=text),
    )


def test_get_reads_dicts_and_objects_first_non_none():
    assert get({"a": None, "b": 2}, "a", "b") == 2
    assert get(SimpleNamespace(a=1), "a") == 1
    assert get(None, "a") is None
    assert get({}, "a") is None


def test_count_coerces_and_clamps():
    assert count({"n": 3.0}, "n") == 3
    assert count({"n": -2}, "n") == 0
    assert count({"n": True}, "n") == 0
    assert count({"n": "7"}, "n") == 0
    assert count({}, "n") == 0


def test_dump_models_and_mappings():
    class Item(BaseModel):
        a: int
        b: str | None = None

    assert dump(Item(a=1)) == {"a": 1}
    source = {"x": 1}
    copied = dump(source)
    assert copied == source and copied is not source


def test_items_lists_only():
    assert items({"xs": (1, 2)}, "xs") == [1, 2]
    assert items({"xs": "ab"}, "xs") == []
    assert items({}, "xs") == []


def test_loads_arguments():
    assert loads_arguments("", provider="p", model="m") == {}
    assert loads_arguments(None, provider="p", model="m") == {}
    assert loads_arguments('{"a": 1}', provider="p", model="m") == {"a": 1}
    with pytest.raises(ModelOutputInvalid, match="not valid JSON") as info:
        loads_arguments("{", provider="p", model="m")
    assert info.value.provider == "p" and isinstance(info.value.__cause__, ValueError)
    with pytest.raises(ModelOutputInvalid, match="not a JSON object"):
        loads_arguments("[1]", provider="p", model="m")


def test_dumps_arguments_compact_unicode():
    assert dumps_arguments({"city": "Zürich", "n": 1}) == '{"city":"Zürich","n":1}'


def test_with_parsed_parses_stop_replies_only():
    assert with_parsed(_reply('{"a": 1}'), SCHEMA).parsed == {"a": 1}
    assert with_parsed(_reply('{"a": 1}'), None).parsed is None
    assert with_parsed(_reply("{", finish="length"), SCHEMA).parsed is None
    called = _reply("", finish="stop", calls=(ToolCall(name="t"),))
    assert with_parsed(called, SCHEMA).parsed is None
    assert with_parsed(_reply("ignored"), SCHEMA, sdk_parsed={"b": 2}).parsed == {"b": 2}


def test_with_parsed_invalid_json_raises_output_invalid(caplog):
    with caplog.at_level(logging.WARNING), pytest.raises(ModelOutputInvalid) as info:
        with_parsed(_reply("not json"), SCHEMA)
    assert info.value.request_id == "r1"
    assert "Answer" in str(info.value)
    assert "not json" not in caplog.text  # content stays out of the logs


def test_tool_payload_detects_error_shape():
    assert tool_payload('{"error": {"kind": "x"}}') == ({"error": {"kind": "x"}}, True)
    assert tool_payload('{"error": 1, "value": 2}') == ({"error": 1, "value": 2}, False)
    assert tool_payload("plain text") == ("plain text", False)
    assert tool_payload("[1, 2]") == ([1, 2], False)


def test_tool_names_from_assistant_calls():
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", tool_calls=(ToolCall(name="a", id="1"), ToolCall(name="b"))),
    ]
    assert tool_names(history) == {"1": "a"}


def test_warn_unsupported_logs_once(caplog, monkeypatch):
    monkeypatch.setattr(_convert, "_warned", set())
    with caplog.at_level(logging.WARNING):
        warn_unsupported("openai", "seed", "responses API")
        warn_unsupported("openai", "seed", "responses API")
    assert caplog.text.count("'seed'") == 1
