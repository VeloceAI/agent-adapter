"""Approval memory in chat and the Runs desk: context shown, and a person still decides."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.approval import ApprovalDecision, ApprovalRequest, SQLiteApprovalStore
from agent_adapter.viewer import ApprovalDesk
from agent_adapter.viewer.chat import ChatMessage, ChatSession
from agent_adapter.viewer.chat.approval_history import arguments_of, context_view, record_of
from tests.viewer.chat.helpers import (
    chat,
    chat_served,
    new_session,
    pending_approval,
    post,
    read_stream,
    send_turn,
    sent,
    tools,
)


async def sender(message: ChatMessage, session: ChatSession) -> str:
    """Sends to the address in the message; the send tool needs approval."""
    try:
        result = await session.executor.run(
            "send", {"to": message.text, "api_key": "sk-0123456789abcdef"}, session.context()
        )
    except Exception as exc:
        return f"not sent: {type(exc).__name__}"
    return str(result)


def decide(server: Any, approval_id: str, approved: bool) -> None:
    status, body = post(server, f"/api/chat/approvals/{approval_id}/decide", {"approved": approved})
    assert status == 200, body


def test_a_chat_approval_shows_earlier_decisions_and_still_waits_for_a_person(
    tmp_path: Path,
) -> None:
    sent.clear()
    with chat_served(tmp_path, agent=sender, registry=tools) as server:
        sid = new_session(server)
        for approved in (True, True, False):
            turn = send_turn(server, sid, "a@example.com")
            decide(server, pending_approval(server, turn["turn_id"])["approval_id"], approved)
            read_stream(server, turn["turn_id"])
        turn = send_turn(server, sid, "a@example.com")
        asked = pending_approval(server, turn["turn_id"])
        context = asked["context"]
        assert (context["similar"], context["approved"], context["denied"]) == (3, 2, 1)
        assert context["deciders"] == ["local"]
        assert context["examples"][0]["matched"] == ["to"]
        assert "sk-0123456789abcdef" not in repr(context) and "api_key" not in repr(context)
        assert len(sent) == 2, "history never ran the fourth call"
        live = chat(server).live(turn["turn_id"])
        assert asked["status"] == "pending" and not live.ended
        decide(server, asked["approval_id"], False)
        events = read_stream(server, turn["turn_id"])
    assert len(sent) == 2 and isinstance(events, list)
    assert "not sent" in "".join(e["text"] for e in events if e["type"] == "text")


def test_context_survives_a_restart_from_the_stored_turns(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=sender, registry=tools) as server:
        sid = new_session(server)
        turn = send_turn(server, sid, "b@example.com")
        decide(server, pending_approval(server, turn["turn_id"])["approval_id"], True)
        read_stream(server, turn["turn_id"])
    with chat_served(tmp_path, agent=sender, registry=tools) as server:
        turn = send_turn(server, sid, "b@example.com")
        asked = pending_approval(server, turn["turn_id"])
        assert asked["context"]["similar"] == 1 and asked["status"] == "pending"
        decide(server, asked["approval_id"], False)
        read_stream(server, turn["turn_id"])


def test_a_memory_failure_asks_without_context(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def broken() -> list[dict[str, Any]]:
        raise RuntimeError("store unreadable")

    with chat_served(tmp_path, agent=sender, registry=tools) as server:
        chat(server).desk.history = broken
        turn = send_turn(server, new_session(server), "c@example.com")
        with caplog.at_level(logging.WARNING):
            asked = pending_approval(server, turn["turn_id"])
        assert asked["context"] is None and asked["status"] == "pending"
        decide(server, asked["approval_id"], False)
        read_stream(server, turn["turn_id"])
    assert "approval memory failed" in caplog.text


def test_record_of_keeps_only_decisions_by_a_person() -> None:
    view = {"approval_id": "a", "tool": "send", "arguments": '{"to": "x"}', "created_at": 5}
    assert record_of({**view, "status": "timed-out", "decided_by": ""}) is None
    assert record_of({**view, "status": "approved", "decided_by": ""}) is None
    record = record_of({**view, "status": "denied", "decided_by": "ann"})
    assert record is not None and record.request.arguments == {"to": "x"}
    assert record.decision is not None and record.decision.decided_at == 5.0
    assert record_of({**view, "status": "approved", "decided_by": "ann", "arguments": "[NaN]"})
    bad = {**view, "status": "approved", "decided_by": "ann", "arguments": {"x": float("nan")}}
    assert record_of(bad) is None


def test_arguments_of_reads_only_whole_json_objects() -> None:
    assert arguments_of({"a": 1}) == {"a": 1}
    assert arguments_of('{"a": 1}') == {"a": 1}
    assert arguments_of('{"a": 1}... [+20 chars]') == {}
    assert arguments_of("[1]") == {}
    assert arguments_of(None) == {}


def test_context_view_counts_each_approval_once() -> None:
    view = {"approval_id": "a", "tool": "send", "arguments": "{}", "status": "approved"}
    view["decided_by"] = "ann"
    request = ApprovalRequest(approval_id="new", run_id="r", call_id="c", tool="send")
    assert context_view(request, [view, view])["similar"] == 1


def test_the_runs_desk_lists_pending_approvals_with_their_context(tmp_path: Path) -> None:
    db = tmp_path / "approvals.db"
    store = SQLiteApprovalStore(db)

    def park(approval_id: str) -> None:
        request = ApprovalRequest(
            approval_id=approval_id,
            run_id="run_1",
            call_id=approval_id,
            tool="refund",
            arguments={"order": "o-1", "api_key": "sk-secret-value"},
        )
        asyncio.run(store.create(request))

    park("apr_old")
    asyncio.run(store.decide("apr_old", ApprovalDecision(approved=True, decided_by="ann")))
    park("apr_new")
    desk = ApprovalDesk(db, "bob")
    (pending,) = desk.pending()
    assert pending["approval_id"] == "apr_new" and pending["status"] == "pending"
    assert pending["context"]["similar"] == 1 and pending["context"]["deciders"] == ["ann"]
    assert "sk-secret-value" not in repr(pending)
    assert asyncio.run(store.get("apr_new")).status == "pending"


def test_the_runs_desk_with_nothing_pending_reads_no_history(tmp_path: Path) -> None:
    db = tmp_path / "approvals.db"
    assert asyncio.run(SQLiteApprovalStore(db).list()) == []
    assert ApprovalDesk(db, "bob").pending() == []
