import json
import logging

import pytest
from pydantic import BaseModel

from agent_adapter.cache import Envelope, decode, encode


class Hit(BaseModel):
    title: str


def test_round_trip(monkeypatch):
    monkeypatch.setattr("agent_adapter.cache.codec.time.time", lambda: 100.0)
    data = encode("tool", {"hits": [Hit(title="a")], "n": 1})
    assert json.loads(data) == {
        "v": 1,
        "kind": "tool",
        "stored_at": 100.0,
        "payload": {"hits": [{"title": "a"}], "n": 1},
    }
    assert decode(data, "tool") == Envelope(
        kind="tool", stored_at=100.0, payload={"hits": [{"title": "a"}], "n": 1}
    )


def test_unicode_is_kept():
    assert decode(encode("llm", "héllo"), "llm").payload == "héllo"


@pytest.mark.parametrize("payload", [object(), float("nan"), {"b": b"x"}])
def test_encode_rejects_non_json_payloads(payload):
    with pytest.raises(TypeError):
        encode("tool", payload)


@pytest.mark.parametrize(
    "data",
    [
        b"\xff\xfe",
        b"not json",
        b"[1, 2]",
        b'{"v": 2, "kind": "tool", "stored_at": 1, "payload": 1}',
        b'{"kind": "tool", "stored_at": 1, "payload": 1}',
        b'{"v": 1, "kind": "tool", "stored_at": "now", "payload": 1}',
        b'{"v": 1, "kind": "tool", "stored_at": true, "payload": 1}',
        b'{"v": 1, "kind": "tool", "stored_at": 1}',
    ],
)
def test_decode_returns_none_for_bad_entries(data, caplog):
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.cache.codec"):
        assert decode(data, "tool") is None
    assert caplog.records


def test_decode_rejects_another_kind():
    assert decode(encode("llm", 1), "tool") is None
    assert decode(encode("gcache", "name"), "gcache").payload == "name"
