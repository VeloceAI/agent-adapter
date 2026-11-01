"""Built-in mode end to end: a fake model, tools through the executor, streaming, attachments."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from agent_adapter.core.messages import LLMRequest, StreamDone, StreamEvent, TextDelta
from agent_adapter.core.types import Usage
from agent_adapter.models.callable import CallableModel
from agent_adapter.viewer.chat.setup import DEFAULT_MAX_CALLS
from tests.fakes.models import ScriptedModel, calls, text
from tests.viewer.chat.helpers import (
    chat,
    chat_served,
    get,
    new_session,
    post,
    read_stream,
    send_turn,
    tools,
)
from tests.viewer.helpers import refreshed, request

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def upload(server, sid: str, name: str, data: bytes) -> tuple[int, dict]:
    headers = {
        "Origin": f"http://127.0.0.1:{server.port}",
        "X-AA-CSRF": server.csrf,
        "Content-Type": "application/octet-stream",
        "X-AA-Filename": name,
    }
    status, body, _ = request(
        server, "POST", f"/api/chat/files?session={sid}", body=data, headers=headers
    )
    return status, body


def test_builtin_mode_runs_tools_and_reports_usage_and_records_the_run(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            calls(("add", {"a": 2, "b": 3}), usage=Usage(input_tokens=10, output_tokens=2)),
            text("it is 5", usage=Usage(input_tokens=12, output_tokens=3)),
        ]
    )
    with chat_served(tmp_path, model=model, registry=tools, system="be brief") as server:
        assert get(server, "/api/chat")[1] == {
            "enabled": True,
            "mode": "builtin",
            "agent_name": "scripted:scripted",
        }
        assert chat(server).executor.limits.max_calls == DEFAULT_MAX_CALLS
        sid = new_session(server)
        turn = send_turn(server, sid, "what is 2+3?")
        events = read_stream(server, turn["turn_id"])
        detail = get(server, f"/api/chat/sessions/{sid}")[1]
    assert "".join(e["text"] for e in events if e["type"] == "text") == "it is 5"
    final_tool = [e for e in events if e["type"] == "tool"][-1]
    assert final_tool["name"] == "add" and final_tool["status"] == "ok"
    assert final_tool["arguments"] == '{"a": 2, "b": 3}' and final_tool["output"] == "5"
    assert isinstance(final_tool["ms"], float)
    usage = [e for e in events if e["type"] == "usage"][-1]
    assert (usage["input_tokens"], usage["output_tokens"]) == (22, 5)
    stored = detail["turns"][0]
    assert stored["status"] == "done" and stored["run_id"] == turn["run_id"]
    assert stored["input_tokens"] == 22 and stored["tools"][0]["name"] == "add"
    assert model.requests[0].messages[0].content == "be brief"
    runs = refreshed(tmp_path / "runs").summaries()
    assert [run["run_id"] for run in runs] == [turn["run_id"]]


def test_the_second_turn_sends_the_first_as_history(tmp_path: Path) -> None:
    model = ScriptedModel([text("first answer"), text("second answer")])
    with chat_served(tmp_path, model=model, registry=tools) as server:
        sid = new_session(server)
        read_stream(server, send_turn(server, sid, "one")["turn_id"])
        read_stream(server, send_turn(server, sid, "two")["turn_id"])
    roles = [(m.role, m.content) for m in model.requests[1].messages]
    assert roles == [
        ("user", "one"),
        ("assistant", "first answer"),
        ("user", "two"),
    ]


def test_a_streaming_model_sends_each_chunk(tmp_path: Path) -> None:
    async def events(request: LLMRequest) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text="hel")
        yield TextDelta(text="lo")
        yield StreamDone(response=text("hello"))

    model = CallableModel(lambda request: "unused", stream_fn=events)
    with chat_served(tmp_path, model=model, registry=tools) as server:
        found = read_stream(server, send_turn(server, new_session(server), "hi")["turn_id"])
    assert [e["text"] for e in found if e["type"] == "text"] == ["hel", "lo"]


def test_text_attachments_are_inlined_and_other_types_refused(tmp_path: Path) -> None:
    model = ScriptedModel([text("read it")])
    with chat_served(tmp_path, model=model, registry=tools) as server:
        sid = new_session(server)
        status, note = upload(server, sid, "notes.md", b"# heading\nbody")
        assert status == 201 and note["media_type"] == "text/markdown"
        status, image = upload(server, sid, "pic.png", PNG)
        assert status == 201 and image["media_type"] == "image/png"
        status, refused = post(
            server,
            f"/api/chat/sessions/{sid}/turns",
            {"text": "see", "attachment_ids": [image["attachment_id"]]},
        )
        assert status == 400 and "--agent" in refused["error"]
        turn = send_turn(server, sid, "see", attachment_ids=[note["attachment_id"]])
        read_stream(server, turn["turn_id"])
    sent = model.requests[0].messages[-1].content
    assert sent == "see\n\n--- attachment: notes.md ---\n# heading\nbody"


def test_a_tool_loop_that_hits_max_steps_says_so(tmp_path: Path) -> None:
    model = ScriptedModel([calls(("add", {"a": 1, "b": 1}))])
    with chat_served(tmp_path, model=model, registry=tools, max_steps=1) as server:
        found = read_stream(server, send_turn(server, new_session(server), "go")["turn_id"])
    assert "[the tool loop ended: max_steps]" in found[-1]["turn"]["answer"]
