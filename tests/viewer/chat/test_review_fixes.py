"""Regressions from the chat backend review: bad Unicode, deep JSON, masks, recorders, CLI."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent_adapter.core.errors import ToolError
from agent_adapter.core.messages import LLMRequest, StreamEvent
from agent_adapter.exec import Executor
from agent_adapter.viewer import record
from agent_adapter.viewer.chat import ChatMessage, ChatSession
from agent_adapter.viewer.chat import host as host_module
from agent_adapter.viewer.chat import stream as stream_module
from agent_adapter.viewer.chat.files import display_name
from agent_adapter.viewer.chat.live import preview
from agent_adapter.viewer.chat.stream import frame
from tests.fakes.models import ScriptedModel, text
from tests.viewer.chat.helpers import (
    chat,
    chat_served,
    get,
    new_session,
    pending_approval,
    post,
    read_stream,
    send_turn,
    sent,
    tools,
)
from tests.viewer.chat.test_contract import async_text
from tests.viewer.helpers import decide_headers, refreshed, request

LONE = chr(0xD83D)  # half of a surrogate pair: valid in JSON, not encodable as UTF-8


def raw_post(server, path: str, raw: bytes) -> tuple[int, dict]:
    status, body, _ = request(server, "POST", path, body=raw, headers=decide_headers(server))
    return status, body


# --- H1: lone surrogates --------------------------------------------------------------------


def test_a_lone_surrogate_in_input_is_refused_and_the_session_stays_usable(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server)
        bad = b'{"text": "hi \\ud83d"}'
        status, body = raw_post(server, f"/api/chat/sessions/{sid}/turns", bad)
        assert status == 400 and "Unicode" in body["error"]
        assert (
            raw_post(server, f"/api/chat/sessions/{sid}/state", b'{"state": {"a": "\\ud83d"}}')[0]
            == 400
        )
        assert raw_post(server, "/api/chat/sessions", b'{"title": "\\udc00"}')[0] == 400
        assert get(server, f"/api/chat/sessions/{sid}")[0] == 200


async def surrogate_agent(message: ChatMessage, session: ChatSession) -> AsyncIterator[str]:
    session.state["note"] = f"x{LONE}y"
    yield f"half {LONE} emoji"


def test_agent_text_and_state_with_a_lone_surrogate_still_save_and_stream(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=surrogate_agent) as server:
        sid = new_session(server)
        turn = send_turn(server, sid, "go")
        events = read_stream(server, turn["turn_id"])
        status, detail = get(server, f"/api/chat/sessions/{sid}")
    assert events[-1]["type"] == "done" and events[-1]["turn"]["answer"] == "half ? emoji"
    assert status == 200
    with chat_served(tmp_path, agent=surrogate_agent) as server:  # the file was saved
        again = get(server, f"/api/chat/sessions/{sid}")[1]
    assert again["turns"][0]["answer"] == "half ? emoji" and "note" in again["state"]
    assert detail["state"]["note"] == again["state"]["note"]


def test_an_error_text_with_a_lone_surrogate_still_ends_the_stream(tmp_path: Path) -> None:
    async def failing(message: ChatMessage, session: ChatSession) -> str:
        raise ToolError(f"bad {LONE}")

    with chat_served(tmp_path, agent=failing) as server:
        sid = new_session(server)
        events = read_stream(server, send_turn(server, sid, "x")["turn_id"])
        assert get(server, f"/api/chat/sessions/{sid}")[0] == 200
    assert events[-1]["type"] == "error" and events[-1]["kind"] == "tool-error"


def test_frames_and_previews_are_always_encodable() -> None:
    raw = frame({"seq": 0, "type": "text", "text": f"a{LONE}\u2028"})
    assert raw.isascii() and json.loads(raw.decode().split("data: ")[1]) is not None
    assert preview(f"x{LONE}") == "x?"


# --- M1: deep JSON, failures, consistent state -------------------------------------------------


def test_deep_json_is_a_400_not_a_dropped_connection(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server, state={"keep": 1})
        status, body = raw_post(server, "/api/chat/sessions", b"[" * 30_000 + b"]" * 30_000)
        assert status == 400 and "nest" in body["error"]
        deep = '{"state": ' + '{"a": ' * 100 + "1" + "}" * 100 + "}"
        status, body = raw_post(server, f"/api/chat/sessions/{sid}/state", deep.encode())
        assert status == 400 and "64 levels" in body["error"]
        assert get(server, f"/api/chat/sessions/{sid}")[1]["state"] == {"keep": 1}


def test_an_unexpected_error_in_a_route_is_a_logged_json_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret detail")

    with chat_served(tmp_path, agent=async_text) as server:
        store = chat(server).store
        monkeypatch.setattr(store, "summaries", broken)
        monkeypatch.setattr(store, "create", broken)
        with caplog.at_level(logging.ERROR):
            assert get(server, "/api/chat/sessions") == (500, {"error": "the request failed"})
            assert post(server, "/api/chat/sessions", {}) == (500, {"error": "the request failed"})
    assert "RuntimeError" in caplog.text and "secret detail" not in caplog.text


def test_a_failed_state_copy_changes_neither_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server, state={"n": 1})
        session = chat(server).store.get(sid)

        def broken(value: object) -> object:
            raise RecursionError("too deep")

        monkeypatch.setattr(host_module.copy, "deepcopy", broken)
        assert post(server, f"/api/chat/sessions/{sid}/state", {"state": {"n": 2}})[0] == 500
        assert session.state == session.saved_state == {"n": 1}


# --- M2: masked values posted back ---------------------------------------------------------


def test_masked_state_posted_back_keeps_every_secret(tmp_path: Path) -> None:
    seen: list[dict] = []

    async def remember(message: ChatMessage, session: ChatSession) -> str:
        seen.append(json.loads(json.dumps(session.state)))
        return "ok"

    secret = "sk-abcdefghijklmnopqrstuvwxyz0123"
    state = {"key": secret, "profiles": [{"api_key": "k-1", "name": "a"}], "n": 1}
    with chat_served(tmp_path, agent=remember) as server:
        sid = new_session(server, state=state)
        shown = get(server, f"/api/chat/sessions/{sid}")[1]["state"]
        assert shown == {"key": "***", "profiles": [{"api_key": "***", "name": "a"}], "n": 1}
        shown["n"] = 2
        assert post(server, f"/api/chat/sessions/{sid}/state", {"state": shown})[0] == 200
        read_stream(server, send_turn(server, sid, "go")["turn_id"])
    assert seen == [{**state, "n": 2}]


@pytest.mark.parametrize(
    "posted",
    [
        {"profiles": [{"api_key": "***"}, {"api_key": "***"}]},  # the list changed length
        {"password": "***"},  # nothing stored under it
    ],
)
def test_a_mask_that_matches_no_stored_value_is_refused(tmp_path: Path, posted: dict) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server, state={"profiles": [{"api_key": "k-1"}]})
        status, body = post(server, f"/api/chat/sessions/{sid}/state", {"state": posted})
        assert status == 400 and "type the secret again" in body["error"]
        assert chat(server).store.get(sid).state == {"profiles": [{"api_key": "k-1"}]}


def test_a_state_post_with_an_unmatched_mask_applies_none_of_its_edits(tmp_path: Path) -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz0123"
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server, state={"key": secret, "n": 1})
        # The old code stored this: "n" changed and "password" became a literal "***".
        posted = {"key": "***", "n": 2, "password": "***"}
        status, _ = post(server, f"/api/chat/sessions/{sid}/state", {"state": posted})
        session = chat(server).store.get(sid)
        assert status == 400
        assert session.state == session.saved_state == {"key": secret, "n": 1}
    stored = json.loads((tmp_path / "chats" / f"{sid}.json").read_text(encoding="utf-8"))
    assert stored["state"] == {"key": secret, "n": 1}


# --- M3, M4: an executor already recorded; tools with --agent ----------------------------------


def test_an_executor_already_recorded_to_the_runs_folder_still_starts_chat(tmp_path: Path) -> None:
    executor = Executor(tools)
    recorder = record(executor, folder=tmp_path / "runs")
    try:
        with chat_served(tmp_path, agent=async_text, executor=executor) as server:
            turn = send_turn(server, new_session(server), "x")
            read_stream(server, turn["turn_id"])
    finally:
        recorder.stop()
    assert turn["run_id"] in [r["run_id"] for r in refreshed(tmp_path / "runs").summaries()]


async def uses_tools(message: ChatMessage, session: ChatSession) -> str:
    result = await session.executor.run("send", {"to": "bob"}, ctx=session.context())
    return str(result)


def test_agent_mode_tools_run_through_session_executor_with_approvals(tmp_path: Path) -> None:
    sent.clear()
    with chat_served(tmp_path, agent=uses_tools, registry=tools) as server:
        turn = send_turn(server, new_session(server), "go")
        asked = pending_approval(server, turn["turn_id"])
        path = f"/api/chat/approvals/{asked['approval_id']}/decide"
        assert post(server, path, {"approved": True})[0] == 200
        events = read_stream(server, turn["turn_id"])
    assert sent == ["bob"] and events[-1]["turn"]["answer"] == "sent to bob"
    assert [e for e in events if e["type"] == "tool"][-1]["status"] == "ok"


def test_ui_accepts_tools_with_an_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.viewer.chat.test_setup import no_serving, run_cli

    served = no_serving(monkeypatch)
    code, _ = run_cli(
        "ui",
        str(tmp_path / "runs"),
        "--port",
        "0",
        "--chat-dir",
        str(tmp_path / "chats"),
        "--agent",
        "tests.viewer.chat.test_review_fixes:uses_tools",
        "--tools",
        "tests.viewer.chat.helpers:tools",
    )
    host = served[0].chat
    assert code == 0 and host is not None and host.executor.registry is tools


# --- lows ------------------------------------------------------------------------------------


def test_a_stop_before_the_task_runs_ends_the_turn_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def agent(message: ChatMessage, session: ChatSession) -> str:
        calls.append(message.text)
        return "ran"

    spawn = host_module.ChatHost._spawn

    async def stop_first(self, live, session, turn, attachments):  # type: ignore[no-untyped-def]
        await self._cancel(live, "stopped")  # a stop that lands before the task exists
        await spawn(self, live, session, turn, attachments)

    monkeypatch.setattr(host_module.ChatHost, "_spawn", stop_first)
    with chat_served(tmp_path, agent=agent) as server:
        events = read_stream(server, send_turn(server, new_session(server), "x")["turn_id"])
    assert calls == [] and events[-1]["status"] == "stopped"


def test_a_refused_edit_leaves_no_empty_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server)
        first = send_turn(server, sid, "a")
        read_stream(server, first["turn_id"])
        monkeypatch.setattr(host_module, "MAX_RUNNING", 0)
        monkeypatch.setattr(host_module.ChatHost, "_check_room", _room_once())
        body = {"text": "b", "edit_of": first["turn_id"]}
        assert post(server, f"/api/chat/sessions/{sid}/turns", body)[0] == 409
        assert [s["id"] for s in get(server, "/api/chat/sessions")[1]["sessions"]] == [sid]
    assert sorted(p.name for p in (tmp_path / "chats").glob("ses_*.json")) == [f"{sid}.json"]


def _room_once():  # type: ignore[no-untyped-def]
    """A room check that passes before the fork and refuses after it."""
    original = host_module.ChatHost._check_room

    def check(self, *, held: bool = False) -> None:  # type: ignore[no-untyped-def]
        if held:
            original(self, held=held)

    return check


def test_a_resume_past_a_stored_turn_still_gets_its_end_event(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        turn = send_turn(server, new_session(server), "x")
        read_stream(server, turn["turn_id"])
    with chat_served(tmp_path, agent=async_text) as server:
        events = read_stream(server, turn["turn_id"], since=9)
    assert [e["type"] for e in events] == ["done"]


def test_a_json_string_tool_output_is_masked() -> None:
    assert preview('{"api_key": "k-1", "n": 1}') == '{"api_key": "***", "n": 1}'
    assert preview("[not json") == "[not json"


def test_the_builtin_loop_can_skip_streaming(tmp_path: Path) -> None:
    class NoStream(ScriptedModel):
        async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
            raise AssertionError("stream must not be called")
            yield  # pragma: no cover

    with chat_served(
        tmp_path, model=NoStream([text("plain")]), registry=tools, stream=False
    ) as server:
        events = read_stream(server, send_turn(server, new_session(server), "x")["turn_id"])
    assert events[-1]["turn"]["answer"] == "plain"


def test_a_stream_error_after_the_head_is_logged_and_ends_the_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(event: dict) -> bytes:
        raise RuntimeError("frame failed")

    with chat_served(tmp_path, agent=async_text) as server:
        turn = send_turn(server, new_session(server), "x")
        monkeypatch.setattr(stream_module, "frame", broken)
        with caplog.at_level(logging.ERROR):
            raw = read_stream(server, turn["turn_id"], raw=True)
    assert raw == "" and "stream of" in caplog.text


def test_names_lose_format_characters() -> None:
    assert display_name("invoice\u202egpj.exe") == "invoicegpj.exe"


def test_close_with_a_turn_waiting_on_approval_stops_it(tmp_path: Path) -> None:
    sent.clear()
    with chat_served(tmp_path, agent=uses_tools, registry=tools) as server:
        turn = send_turn(server, new_session(server), "go")
        pending_approval(server, turn["turn_id"])
        chat(server).close()
        events = read_stream(server, turn["turn_id"])
    assert sent == []
    assert events[-1]["status"] == "stopped"
    assert [e for e in events if e["type"] == "approval"][-1]["status"] == "stopped"


def test_an_endless_timeout_is_refused(tmp_path: Path) -> None:
    from agent_adapter.core.errors import ConfigError
    from agent_adapter.viewer.chat.setup import build_host

    with pytest.raises(ConfigError, match="turn_timeout"):
        build_host(runs=tmp_path, agent=async_text, chat_dir=tmp_path, turn_timeout=float("inf"))
