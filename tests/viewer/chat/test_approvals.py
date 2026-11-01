"""Approvals in chat: one-use decisions, masking, stop and timeouts deny, failures deny."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_adapter.approval import AutoApprove
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.viewer.chat.approvals import ChatApprovals
from agent_adapter.viewer.chat.live import LiveTurn
from agent_adapter.viewer.chat.watch import TurnWatch
from tests.fakes.models import ScriptedModel, calls, text
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


@pytest.fixture(autouse=True)
def clear_sent() -> None:
    sent.clear()


def sender() -> ScriptedModel:
    return ScriptedModel(
        [calls(("send", {"to": "bob", "api_key": "sk-123"})), text("done sending")]
    )


def decide(server, approval_id: str, approved: bool, reason: str = "") -> tuple[int, dict]:
    return post(
        server,
        f"/api/chat/approvals/{approval_id}/decide",
        {"approved": approved, "reason": reason},
    )


def test_an_approved_call_runs_once_and_a_second_decision_is_409(tmp_path: Path) -> None:
    with chat_served(tmp_path, model=sender(), registry=tools, reviewer="ana") as server:
        turn = send_turn(server, new_session(server), "send it")
        asked = pending_approval(server, turn["turn_id"])
        assert asked["tool"] == "send" and '"api_key": "***"' in asked["arguments"]
        assert "sk-123" not in asked["arguments"]
        status, body = decide(server, asked["approval_id"], True, "fine")
        assert status == 200 and body["approval"]["status"] == "approved"
        assert body["approval"]["decided_by"] == "ana"
        assert decide(server, asked["approval_id"], False) == (409, {"error": "already approved"})
        events = read_stream(server, turn["turn_id"])
    assert sent == ["bob"]
    assert events[-1]["turn"]["approvals"][0]["status"] == "approved"
    assert "sk-123" not in str(events)


def test_a_denied_call_does_not_run(tmp_path: Path) -> None:
    with chat_served(tmp_path, model=sender(), registry=tools) as server:
        turn = send_turn(server, new_session(server), "send it")
        asked = pending_approval(server, turn["turn_id"])
        assert decide(server, asked["approval_id"], False, "no")[0] == 200
        events = read_stream(server, turn["turn_id"])
    assert sent == []
    tool = [e for e in events if e["type"] == "tool"][-1]
    assert tool["status"] == "failed" and tool["error_kind"] == "approval-denied"


def test_stop_denies_the_pending_approval_and_ends_the_turn(tmp_path: Path) -> None:
    with chat_served(tmp_path, model=sender(), registry=tools) as server:
        turn = send_turn(server, new_session(server), "send it")
        asked = pending_approval(server, turn["turn_id"])
        status, body = post(server, f"/api/chat/turns/{turn['turn_id']}/stop", {})
        assert status == 200 and body["stopping"] is True
        events = read_stream(server, turn["turn_id"])
        assert decide(server, asked["approval_id"], True) == (409, {"error": "already stopped"})
        assert post(server, f"/api/chat/turns/{turn['turn_id']}/stop", {})[0] == 409
    assert sent == []
    assert events[-1]["status"] == "stopped" and events[-1]["kind"] == "stopped"
    approvals = [e for e in events if e["type"] == "approval"]
    assert approvals[-1]["status"] == "stopped"


def test_the_approval_timeout_denies(tmp_path: Path) -> None:
    with chat_served(tmp_path, model=sender(), registry=tools, approval_timeout=0.2) as server:
        events = read_stream(server, send_turn(server, new_session(server), "go")["turn_id"])
    assert sent == []
    approvals = [e for e in events if e["type"] == "approval"]
    assert approvals[-1]["status"] == "timed-out" and "0.2 s" in approvals[-1]["reason"]
    assert events[-1]["type"] == "done"


def test_the_turn_timeout_ends_the_turn_and_denies_its_approval(tmp_path: Path) -> None:
    with chat_served(tmp_path, model=sender(), registry=tools, turn_timeout=0.3) as server:
        turn = send_turn(server, new_session(server), "go")
        events = read_stream(server, turn["turn_id"])
        asked = next(e for e in events if e["type"] == "approval")
        assert decide(server, asked["approval_id"], True)[0] == 409
    assert sent == []
    assert (events[-1]["status"], events[-1]["kind"]) == ("timed-out", "turn-timeout")
    assert [e for e in events if e["type"] == "approval"][-1]["status"] == "timed-out"


def test_decide_refuses_unknown_ids_and_bad_bodies(tmp_path: Path) -> None:
    with chat_served(tmp_path, model=sender(), registry=tools) as server:
        assert decide(server, "apr_unknown", True) == (404, {"error": "no such approval"})
        status, body = post(server, "/api/chat/approvals/apr_x/decide", {"approved": "yes"})
        assert status == 400 and "approved" in body["error"]


def test_an_executor_with_its_own_approval_port_keeps_it(tmp_path: Path) -> None:
    executor = Executor(tools, approval=AutoApprove())
    with chat_served(tmp_path, model=sender(), executor=executor) as server:
        events = read_stream(server, send_turn(server, new_session(server), "go")["turn_id"])
        assert chat(server).executor is not executor  # a copy with the watch added
    assert sent == ["bob"] and not [e for e in events if e["type"] == "approval"]


def request_for(run_id: str) -> ApprovalRequest:
    return ApprovalRequest(approval_id="apr_1", run_id=run_id, call_id="c1", tool="send")


async def test_a_request_outside_a_chat_turn_is_denied() -> None:
    desk = ChatApprovals(TurnWatch(), reviewer="local", timeout=5)
    decision = await desk.port().request(request_for("run_other"))
    assert not decision.approved and "chat turn" in decision.reason


async def test_any_failure_in_the_desk_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    watch = TurnWatch()
    desk = ChatApprovals(watch, reviewer="local", timeout=5)

    def broken(run_id: str) -> LiveTurn:
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(watch, "find", broken)
    decision = await desk.ask(request_for("run_1"))
    assert not decision.approved and decision.reason == "the approval desk failed"


async def test_the_same_approval_id_twice_is_denied() -> None:
    watch = TurnWatch()
    live = LiveTurn("turn_1", "ses_1", "run_1")
    watch.add(live)
    desk = ChatApprovals(watch, reviewer="local", timeout=0.05)
    first = await desk.ask(request_for("run_1"))  # times out
    second = await desk.ask(request_for("run_1"))
    assert not first.approved and not second.approved
    assert second.reason == "this approval was already asked"
