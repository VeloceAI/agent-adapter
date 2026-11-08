"""CallableAuditStore ships signed records one by one; verify_records checks the stream."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.audit import CallableAuditStore, verify_records
from agent_adapter.errors import ConfigError, IntegrationError, ToolUsageError
from tests.audit.helpers import KEY, entry


def shipped(count: int = 2) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    store = CallableAuditStore(sent.append, key=KEY)
    for _ in range(count):
        store.write(entry())
    store.close()
    store.close()
    return sent


def test_a_stream_is_open_events_seal_and_verifies() -> None:
    sent = shipped(2)
    assert [record["kind"] for record in sent] == ["open", "event", "event", "seal"]
    assert sent[-1]["events"] == 2 and sent[0]["stream"] == sent[-1]["stream"]
    result = verify_records(sent, KEY)
    assert result.ok and result.events == 2 and result.sealed


def test_a_store_that_got_nothing_sends_nothing() -> None:
    sent: list[dict[str, Any]] = []
    CallableAuditStore(sent.append, key=KEY).close()
    assert sent == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda s: [s[0], s[2], s[3]], "missing or reordered"),
        (lambda s: [s[0], s[2], s[1], s[3]], "missing or reordered"),
        (lambda s: s[:-1], "no seal"),
        (lambda s: s[1:], "open record"),
        (lambda s: [s[0], {**s[1], "action": "x"}, *s[2:]], "MAC"),
    ],
)
def test_tampered_streams_fail(change: Any, expected: str) -> None:
    result = verify_records(change(shipped(2)), KEY)
    assert not result.ok and any(expected in p for p in result.problems)


def test_a_failing_callable_raises_and_leaves_a_visible_gap() -> None:
    sent: list[dict[str, Any]] = []
    calls = {"n": 0}

    def flaky(record: dict[str, Any]) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("siem down")
        sent.append(record)

    store = CallableAuditStore(flaky, key=KEY)
    with pytest.raises(IntegrationError):
        store.write(entry())
    store.write(entry())
    store.close()
    assert any("missing" in p for p in verify_records(sent, KEY).problems)


def test_bad_arguments_and_closed_writes_are_refused() -> None:
    with pytest.raises(ConfigError):
        CallableAuditStore("nope", key=KEY)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        CallableAuditStore(print, key="short")
    store = CallableAuditStore(lambda record: None, key=KEY)
    store.close()
    with pytest.raises(ToolUsageError):
        store.write(entry())
    assert store.digest("ann") == CallableAuditStore(print, key=KEY).digest("ann")
