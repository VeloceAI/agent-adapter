import hashlib
import logging

import pytest
from pydantic import BaseModel

from agent_adapter.cache import KEY_VERSION, canonical_json, digest, gcache_key, llm_key, tool_key
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.messages import LLMRequest, Message


class Point(BaseModel):
    x: int
    y: int


def test_canonical_json_is_sorted_compact_and_utf8():
    assert canonical_json({"b": 1, "a": [1, "é"]}) == '{"a":[1,"é"],"b":1}'
    assert canonical_json({"p": Point(x=1, y=2), "s": {3, 1}, "t": (1,)}) == (
        '{"p":{"x":1,"y":2},"s":[1,3],"t":[1]}'
    )
    assert canonical_json(frozenset({"b", "a"})) == '["a","b"]'


@pytest.mark.parametrize("value", [object(), {"x": float("nan")}, {"x": float("inf")}, b"raw"])
def test_canonical_json_rejects_non_json_values_with_type_error(value):
    with pytest.raises(TypeError):
        canonical_json(value)


def test_digest_is_32_hex_of_sha256():
    expected = hashlib.sha256(b'{"a":1}').hexdigest()[:32]
    assert digest({"a": 1}) == expected and len(expected) == 32
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_tool_key_formats():
    args = {"q": "x"}
    d = digest(args)
    assert KEY_VERSION == 1
    assert tool_key("aa", "search", args, scope="run", run_id="r1", thread_id="t1") == (
        f"aa:tool:v1:run=r1:search:{d}"
    )
    assert tool_key("aa", "search", args, scope="thread", run_id="r1", thread_id="t1") == (
        f"aa:tool:v1:thread=t1:search:{d}"
    )
    assert tool_key("ns", "search@ab12cd34", args, scope="global", run_id="", thread_id="") == (
        f"ns:tool:v1:global:search@ab12cd34:{d}"
    )


def test_tool_key_is_empty_without_an_owner_id(caplog):
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.cache.keys"):
        assert tool_key("aa", "t", {}, scope="thread", run_id="r", thread_id="") == ""
        assert tool_key("aa", "t", {}, scope="run", run_id="", thread_id="t") == ""
    assert "thread-scoped result not cached without a thread id" in caplog.text


def test_tool_key_rejects_unknown_scopes():
    with pytest.raises(ToolUsageError, match="unknown cache scope"):
        tool_key("aa", "t", {}, scope="session", run_id="r", thread_id="t")  # type: ignore[arg-type]


def request(**changes) -> LLMRequest:
    return LLMRequest(messages=[Message(role="user", content="hi")], **changes)


def test_llm_key_format_and_payload():
    req = request(temperature=0.2)
    payload = req.model_dump(mode="json", exclude={"stream", "stage", "response_cache", "metadata"})
    payload["fields"] = {"plugin": "x"}
    key = llm_key("aa", req, provider="openai", model="gpt", fields={"plugin": "x"})
    assert key == f"aa:llm:v1:openai:gpt:{digest(payload)}"


def test_llm_key_ignores_transport_fields_but_not_content():
    base = llm_key("aa", request(), provider="p", model="m", fields={})
    same = request(stream=True, stage="plan", response_cache=False, metadata={"call_id": "c"})
    assert llm_key("aa", same, provider="p", model="m", fields={}) == base
    assert llm_key("aa", request(temperature=1.0), provider="p", model="m", fields={}) != base
    assert llm_key("aa", request(extra={"k": 1}), provider="p", model="m", fields={}) != base
    assert llm_key("aa", request(), provider="p", model="m", fields={"f": 1}) != base


def test_gcache_key_format():
    assert gcache_key("aa", "gemini-3", "abc") == "aa:gcache:v1:gemini-3:abc"


@pytest.mark.parametrize(
    "changes",
    [
        {"extra": {"client": object()}},
        {"extra": {"temperature": float("nan")}},
        {"extra": {"limit": float("inf")}},
    ],
)
def test_llm_key_rejects_uncacheable_extra_with_type_error(changes):
    with pytest.raises(TypeError):
        llm_key("aa", request(**changes), provider="p", model="m", fields={})


@pytest.mark.parametrize("fields", [{"f": object()}, {"f": float("nan")}])
def test_llm_key_rejects_uncacheable_fields_with_type_error(fields):
    with pytest.raises(TypeError):
        llm_key("aa", request(), provider="p", model="m", fields=fields)


def test_nan_does_not_collide_with_null():
    with pytest.raises(TypeError):
        llm_key("aa", request(temperature=float("nan")), provider="p", model="m", fields={})
    llm_key("aa", request(temperature=None), provider="p", model="m", fields={})
