"""AuditSink on a real executor: what lands, what never lands, attach and detach."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from agent_adapter import AutoApprove, CallbackApproval, Executor, Registry
from agent_adapter.audit import AuditSink, CallableAuditStore, FileAuditStore, audit, read_entries
from agent_adapter.audit._files import segments
from agent_adapter.core.types import Event
from agent_adapter.errors import ApprovalDenied, ConfigError, GuardrailBlocked
from agent_adapter.exec.defaults import default_executor
from agent_adapter.hooks import GuardrailHook
from agent_adapter.tracing.events import emit, make_event
from agent_adapter.tracing.sinks import members
from tests.audit.helpers import KEY

SECRET = "sk-live0123456789abcdefghijklmnop"
EMAIL = "ann@example.com"


def registry() -> Registry:
    reg = Registry()

    @reg.tool(confirm=True)
    def refund(order: str, api_key: str = "") -> str:
        """Refund an order."""
        return f"refunded {order}"

    @reg.tool
    def lookup(order: str) -> str:
        """Look an order up."""
        return order

    return reg


def actions(folder: Path) -> list[str]:
    return [str(record["action"]) for record in read_entries(folder, KEY)]


def log_text(folder: Path) -> str:
    return "".join(path.read_text(encoding="utf-8") for _, path in segments(folder))


def test_approved_calls_are_requested_decided_and_consumed(tmp_path: Path) -> None:
    executor = Executor(registry(), approval=AutoApprove())
    sink = audit(executor, folder=tmp_path, key=KEY, fsync=False)
    asyncio.run(executor.run("refund", {"order": "A-1"}))
    asyncio.run(executor.run("lookup", {"order": "A-2"}))  # not security-relevant
    sink.stop()
    assert actions(tmp_path) == ["approval.requested", "approval.decided", "approval.consumed"]
    assert sink.written == 3 and sink.healthy


def test_denied_and_blocked_calls_are_recorded(tmp_path: Path) -> None:
    def block(stage: str, payload: Any) -> bool | str:
        return "blocked" if payload.get("order") == "bad" else True

    executor = Executor(
        registry(), approval=CallbackApproval(lambda request: False), hooks=[GuardrailHook(block)]
    )
    sink = audit(executor, folder=tmp_path, key=KEY, fsync=False)
    with pytest.raises(ApprovalDenied):
        asyncio.run(executor.run("refund", {"order": "A-1"}))
    with pytest.raises(GuardrailBlocked):
        asyncio.run(executor.run("lookup", {"order": "bad"}))
    sink.stop()
    assert actions(tmp_path) == [
        "approval.requested",
        "approval.decided",
        "approval.denied",
        "guardrail.blocked",
    ]


def test_no_port_is_a_recorded_denial(tmp_path: Path) -> None:
    executor = Executor(registry())
    sink = audit(executor, folder=tmp_path, key=KEY, fsync=False)
    with pytest.raises(ApprovalDenied):
        asyncio.run(executor.run("refund", {"order": "A-1"}))
    sink.stop()
    assert actions(tmp_path) == ["approval.denied"]


def test_secrets_pii_and_content_never_land(tmp_path: Path) -> None:
    executor = Executor(registry(), approval=AutoApprove())
    sink = audit(executor, folder=tmp_path, key=KEY, fsync=False)
    asyncio.run(executor.run("refund", {"order": EMAIL, "api_key": SECRET}))
    ctx = executor.context(run_id="run_mem")
    content: dict[str, Any] = {"arguments": {"text": f"{EMAIL} {SECRET}"}, "result": SECRET}
    for kind in ("memory.added", "memory.deleted", "memory.exported"):
        emit(
            ctx.sink,
            make_event(
                kind, ctx, backend="sqlite", count=1, latency_ms=1, scope_kind="user", **content
            ),
        )
    emit(ctx.sink, make_event("custom.policy.loaded", ctx, rules=2, token=SECRET, owner=EMAIL))
    ctx.emit("audit.override", note=f"by {EMAIL}", password=SECRET)
    sink.stop()
    text = log_text(tmp_path)
    assert SECRET not in text and EMAIL not in text and KEY not in text
    found = actions(tmp_path)
    assert {"memory.added", "memory.deleted", "memory.exported"} <= set(found)
    assert "app.override" in found and "policy.loaded" not in found


def test_a_failing_store_is_counted_and_logged_never_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def down(record: dict[str, Any]) -> None:
        raise RuntimeError(SECRET)

    sink = AuditSink(CallableAuditStore(down, key=KEY))
    with caplog.at_level(logging.ERROR, logger="agent_adapter.audit.sink"):
        sink.emit(Event("approval.requested", data={"tool": "t", "approval_id": "a"}))
        assert sink.flush()
    assert sink.failures == 1 and not sink.healthy
    assert "audit record lost" in caplog.text and SECRET not in caplog.text


def test_a_stopped_sink_ignores_events(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    sink = AuditSink(CallableAuditStore(sent.append, key=KEY))
    sink.stop()
    sink.emit(Event("approval.requested", data={"tool": "t", "approval_id": "a"}))
    assert sent == []


def test_audit_attaches_once_per_folder_and_stop_detaches(tmp_path: Path) -> None:
    executor = Executor(registry(), approval=AutoApprove())
    sink = audit(executor, folder=tmp_path, key=KEY, fsync=False)
    assert audit(executor, folder=tmp_path, key=KEY) is sink
    assert sink in list(members(executor.sink))
    sink.stop()
    assert sink not in list(members(executor.sink))


def test_audit_can_ship_to_a_callable(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    executor = Executor(registry(), approval=AutoApprove())
    sink = audit(executor, store=CallableAuditStore(sent.append, key=KEY))
    asyncio.run(executor.run("refund", {"order": "A-1"}))
    sink.stop()
    assert [record["kind"] for record in sent] == ["open", "event", "event", "event", "seal"]
    json.dumps(sent)  # JSON-safe, ready for any shipper


def test_audit_follows_the_default_executor(tmp_path: Path) -> None:
    sink = audit(folder=tmp_path, key=KEY, fsync=False)
    try:
        assert sink in list(members(default_executor().sink))
    finally:
        sink.stop()
    assert sink not in list(members(default_executor().sink))


def test_bad_stores_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        AuditSink("nope")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        audit(Executor(registry()), store="nope")  # type: ignore[arg-type]


def test_a_store_is_closed_when_attaching_fails(tmp_path: Path) -> None:
    class NotAnExecutor:
        sink = None

        def add_sink(self, sink: object) -> None:
            from agent_adapter.errors import ToolUsageError

            raise ToolUsageError("no")

    with pytest.raises(Exception, match="no"):
        audit(NotAnExecutor(), folder=tmp_path, key=KEY)  # type: ignore[arg-type]
    FileAuditStore(tmp_path, key=KEY).close()  # the folder lock was released
