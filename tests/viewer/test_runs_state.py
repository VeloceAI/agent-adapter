"""State values on node.finished: kept only for traced runs or with content, secrets masked."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from agent_adapter import Executor, Flow, FlowRunner, Registry
from agent_adapter.viewer import Recorder, record
from agent_adapter.viewer._scrub import scrub_data
from agent_adapter.viewer.recorder import flush_all
from tests.viewer.helpers import event, lines, segments

UPDATES = {"answer": "42", "api_key": "sk-live-123", "nested": {"password": "pw", "n": 1}}


def finished(**data: Any) -> Any:
    return event("node.finished", 2.0, "f:n:1", flow="f", node="n", visit=1, **data)


def node_data(folder: Path) -> dict[str, Any]:
    (found,) = segments(folder)
    return next(line for line in lines(found.path) if line.get("type") == "node.finished")


def test_updates_are_left_out_of_an_untraced_run(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    recorder.emit(event("flow.started", 1.0, flow="f"))
    recorder.emit(finished(changed=["answer", "api_key"], updates=UPDATES))
    line = node_data(tmp_path)
    assert line["data"]["changed"] == ["answer", "api_key"]
    assert "updates" not in line["data"] and line["omitted"] == ["updates"]


def test_a_traced_run_keeps_updates_with_secret_names_masked(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    recorder.emit(event("flow.started", 1.0, flow="f", trace_state=True))
    recorder.emit(finished(changed=["answer"], updates=UPDATES))
    data = node_data(tmp_path)["data"]
    assert data["updates"]["answer"] == "42"
    assert data["updates"]["api_key"] == "***"
    assert data["updates"]["nested"] == {"password": "***", "n": 1}


def test_capture_content_keeps_updates_of_any_run(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path, capture_content=True)
    recorder.emit(finished(changed=["answer"], updates={"answer": "x" * 5000}))
    value = node_data(tmp_path)["data"]["updates"]["answer"]
    assert value.endswith("chars]") and len(value) < 2100  # long strings are still clipped


def test_only_a_true_flag_marks_a_run_as_traced(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    recorder.emit(event("flow.started", 1.0, flow="f", trace_state="yes"))
    recorder.emit(finished(changed=["answer"], updates=UPDATES))
    assert "updates" not in node_data(tmp_path)["data"]


def test_scrub_keeps_state_only_when_asked() -> None:
    data = {"updates": {"a": 1}, "arguments": {"q": 1}, "changed": ["a"]}
    assert scrub_data(data, capture_content=False)[1] == ["arguments", "updates"]
    kept, omitted = scrub_data(data, capture_content=False, keep_state=True)
    assert omitted == ["arguments"] and kept["updates"] == {"a": 1}
    assert scrub_data(data, capture_content=True)[1] == []


class Answer(BaseModel):
    question: str = ""
    answer: str = ""


async def test_a_traced_flow_run_shows_its_values_in_the_run_file(tmp_path: Path) -> None:
    def solve(state: Answer) -> dict[str, Any]:
        return {"answer": f"re: {state.question}"}

    executor = Executor(Registry())
    recorder = record(executor, folder=tmp_path)
    flow = Flow("solver", state=Answer).node("solve", solve).end("solve")
    await FlowRunner(executor, trace_state=True).run(flow, {"question": "why"})
    await FlowRunner(executor).run(flow, {"question": "how"})
    recorder.stop()
    found = []
    for segment in segments(tmp_path):
        for line in lines(segment.path):
            if line.get("type") == "node.finished":
                found.append(line["data"])
    traced = [data for data in found if "updates" in data]
    assert [data["updates"] for data in traced] == [{"answer": "re: why"}]
    assert all(data["changed"] == ["answer"] for data in found) and len(found) == 2


def test_a_secret_inside_a_long_traced_value_never_reaches_the_run_file(tmp_path: Path) -> None:
    class Holder(BaseModel):
        config: dict[str, Any] = {}

    def load(state: Holder) -> dict[str, Any]:
        return {"config": {"api_key": "sk-LIVE-123456", "blob": "x" * 3000}}

    executor = Executor(Registry())
    recorder = record(executor, folder=tmp_path)
    flow = Flow("cfg", state=Holder).node("load", load).end("load")
    asyncio.run(FlowRunner(executor, trace_state=True).run(flow, {}))
    recorder.stop()
    (segment,) = segments(tmp_path)
    text = segment.path.read_text(encoding="utf-8")
    assert "sk-LIVE-123456" not in text and "api_key" in text


def test_a_traced_run_is_logged_once_when_content_capture_is_off(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="agent_adapter.viewer._segments")
    recorder = Recorder(tmp_path)
    for ts in (1.0, 1.5):
        recorder.emit(event("flow.started", ts, flow="f", trace_state=True))
    assert flush_all()
    assert caplog.text.count("traces flow state") == 1
