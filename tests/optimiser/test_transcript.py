import json

import pytest

from agent_adapter.core.messages import Message
from agent_adapter.core.types import Event, ToolCall
from agent_adapter.hooks.base import Hook, ShortCircuit
from agent_adapter.optimiser._state import RunState, load_state, save_state
from agent_adapter.optimiser._transcript import (
    TrackedHook,
    TranscriptRecorder,
    hook_metrics,
    render_messages,
)
from tests.optimiser.fakes import reply


class Busy(Hook):
    name = "busy"

    async def before_tool(self, inv):
        return ShortCircuit(1)

    async def after_tool(self, inv, result):
        return None

    async def on_tool_error(self, inv, error):
        raise RuntimeError("no")

    async def before_llm(self, inv):
        return None

    async def after_llm(self, inv, response):
        return response

    async def on_llm_error(self, inv, error):
        return None

    async def before_node(self, inv):
        return ShortCircuit({})

    async def after_node(self, inv, update):
        return None

    async def on_node_error(self, inv, error):
        return None

    def config(self):
        return {"level": 1}


async def test_tracked_hook_delegates_and_counts_actions():
    hook = TrackedHook(Busy())
    assert hook.name == "busy" and hook.config() == {"level": 1}
    assert isinstance(await hook.before_tool(None), ShortCircuit)
    assert await hook.after_tool(None, None) is None
    with pytest.raises(RuntimeError):
        await hook.on_tool_error(None, ValueError())
    assert await hook.before_llm(None) is None
    assert await hook.after_llm(None, "r") == "r"
    assert await hook.on_llm_error(None, ValueError()) is None
    assert isinstance(await hook.before_node(None), ShortCircuit)
    assert await hook.after_node(None, None) is None
    assert await hook.on_node_error(None, ValueError()) is None
    assert hook.acted == 4
    assert hook_metrics([hook]) == {"hook_fired.busy": 4.0}


def test_render_messages_marks_error_observations_only_for_error_objects():
    messages = [
        Message(role="system", content="hidden"),
        Message(role="user", content="q"),
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall(name="t", arguments={"password": "p"}),),
        ),
        Message(role="tool", content=json.dumps({"error": {"kind": "x"}})),
        Message(role="tool", content='{"error": broken'),
        Message(role="tool", content=json.dumps({"error": 1, "value": 2})),
    ]
    text = render_messages(messages)
    assert "hidden" not in text
    assert '[ACTION] t {"password": "***"}' in text
    lines = text.splitlines()
    assert lines[2].startswith("[OBSERVATION ERROR]")
    assert lines[3].startswith("[OBSERVATION] {") and lines[4].startswith("[OBSERVATION] {")


def test_recorder_collects_events_and_replies():
    recorder = TranscriptRecorder(withhold={"pay"})
    recorder.emit(Event(type="node.started", data={"node": "n", "visit": 1}))
    recorder.emit(Event(type="tool.failed", data={"tool": "pay", "error_kind": "approval-denied"}))
    recorder.emit(Event(type="llm.finished", data={}))
    recorder.add_reply("s", "", reply(""))
    assert recorder.text() == ("[NODE] n (visit 1)\n[OBSERVATION ERROR] tool pay: approval-denied")
    assert recorder.counts["tool_calls.pay"] == 1 and recorder.counts["llm_calls"] == 1


def test_state_that_cannot_be_read_or_is_stale_starts_afresh(tmp_path):
    assert load_state(tmp_path, "x") is None
    (tmp_path / "state.json").write_text("{nope", "utf-8")
    assert load_state(tmp_path, "x") is None
    save_state(tmp_path, RunState(fingerprint="old", run_id="opt_1"))
    assert load_state(tmp_path, "new") is None
    assert load_state(tmp_path, "old").run_id == "opt_1"
