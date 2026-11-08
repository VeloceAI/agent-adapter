"""Eval cases from recorded runs: inputs, expected values, tags, filters and skips."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import Event
from agent_adapter.evals import (
    RecordedRun,
    RunCase,
    SkippedRun,
    cases_from_runs,
    load_cases,
    write_cases,
)
from agent_adapter.evals.from_runs import (
    CLIPPED,
    NO_CONTENT,
    NO_INPUT,
    RUNNING,
    UNKNOWN,
    case_from_run,
)
from agent_adapter.viewer import Recorder
from agent_adapter.viewer._tags import TagStore

ASKED = [{"role": "system", "content": "be kind"}, {"role": "user", "content": "Where is 1042?"}]
ANSWER = [{"role": "assistant", "parts": [{"type": "text", "content": "It shipped."}]}]


def ev(etype: str, ts: float, run: str, call: str = "", **data: Any) -> Event:
    return Event(etype, run_id=run, call_id=call, ts=ts, data=data)


def chat_run(run: str, *, ok: bool = True, answer: Any = ANSWER) -> list[Event]:
    return [
        ev("run.started", 1.0, run),
        ev("llm.started", 1.1, run, "l1", provider="p", model="m", input_messages=ASKED),
        ev("tool.started", 1.2, run, "c1", tool="order_status", kind=""),
        ev("tool.finished", 1.3, run, "c1", tool="order_status"),
        ev("llm.finished", 1.4, run, "l1", provider="p", model="m", output_messages=answer),
        ev("run.finished", 2.0, run, ok=ok),
    ]


def recorded(folder: Path, events: list[Event], *, capture: bool = True) -> Path:
    recorder = Recorder(folder, capture_content=capture)
    for item in events:
        recorder.emit(item)
    assert recorder.flush()
    recorder.stop()
    return folder


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    folder = tmp_path / "runs"
    recorded(folder, chat_run("good"))
    recorded(folder, chat_run("bad", ok=False))
    tool_only = [
        ev("flow.started", 3.0, "tool", flow="refunds"),
        ev("tool.started", 3.1, "tool", "t1", tool="refund", kind="", arguments={"order": 7}),
        ev("tool.finished", 3.2, "tool", "t1", tool="refund", result={"refunded": True}),
        ev("run.finished", 3.3, "tool", ok=True),
    ]
    recorded(folder, tool_only)
    recorded(folder, [ev("run.finished", 4.0, "empty", ok=True)])
    TagStore(folder).update("bad", add=["vip"])
    return folder


def by_key(found: Any) -> dict[str, Any]:
    return {item.run_key: item for item in (*found.cases, *found.skipped)}


def test_cases_from_runs_reads_inputs_outputs_and_tags(runs: Path) -> None:
    found = by_key(cases_from_runs(runs))
    good = found["good"]
    assert isinstance(good, RunCase) and good.source_run == "good"
    assert good.case.id == "run-good" and good.case.input == "Where is 1042?"
    assert good.case.expected == "It shipped."
    assert good.case.tags == ("status:ok", "tool:order_status")
    bad = found["bad"]
    assert bad.case.expected is None  # a failed run's output is never the expectation
    assert bad.case.tags == ("vip", "status:failed", "tool:order_status")
    tool = found["tool"]
    assert tool.case.input == {"order": 7} and tool.case.expected == {"refunded": True}
    assert tool.case.tags == ("status:ok", "flow:refunds", "tool:refund")
    assert found["empty"] == SkippedRun("empty", "empty", NO_INPUT)


def test_filters_by_status_tag_and_key(runs: Path) -> None:
    assert [c.run_key for c in cases_from_runs(runs, status="failed").cases] == ["bad"]
    assert [c.run_key for c in cases_from_runs(runs, tags=["VIP"]).cases] == ["bad"]
    picked = cases_from_runs(runs, run_keys=["tool", "gone"], status=["ok", "failed"])
    assert [c.run_key for c in picked.cases] == ["tool"]
    assert picked.skipped == (SkippedRun("gone", "", UNKNOWN),)
    none = cases_from_runs(runs, expected_from="none")
    assert all(c.case.expected is None for c in none.cases)


def test_a_run_without_content_is_skipped_never_invented(tmp_path: Path) -> None:
    folder = recorded(tmp_path / "runs", chat_run("quiet"), capture=False)
    found = cases_from_runs(folder)
    assert found.cases == () and found.skipped == (SkippedRun("quiet", "quiet", NO_CONTENT),)


def test_one_run_file_works_too(runs: Path) -> None:
    one = next(runs.glob("*.good.*.jsonl"))
    assert [c.run_key for c in cases_from_runs(one).cases] == ["good"]


def test_clipped_or_running_runs_are_skipped() -> None:
    long_input = [{"role": "user", "content": "x... [+12 chars]"}]
    events = tuple(e.to_dict() for e in chat_run("r"))
    events = (
        {**events[1], "data": {**events[1]["data"], "input_messages": long_input}},
        *events[2:],
    )
    run = RecordedRun("r", "r", "ok", True, events)
    assert case_from_run(run) == SkippedRun("r", "r", CLIPPED)
    live = RecordedRun("r", "r", "running", True, ())
    assert case_from_run(live) == SkippedRun("r", "r", RUNNING)
    cut_answer = tuple(e.to_dict() for e in chat_run("c", answer=["... +3 items"]))
    made = case_from_run(RecordedRun("c", "c", "ok", True, cut_answer))
    assert isinstance(made, RunCase) and made.case.expected is None


def test_case_from_run_takes_an_id_and_extra_tags() -> None:
    events = tuple(e.to_dict() for e in chat_run("r"))
    made = case_from_run(
        RecordedRun("r", "r", "ok", True, events, ("vip",)),
        case_id="mine",
        extra_tags=["regression", "vip"],
    )
    assert isinstance(made, RunCase)
    assert made.case.id == "mine" and made.case.tags[:2] == ("regression", "vip")
    with pytest.raises(ConfigError, match="expected_from"):
        case_from_run(RecordedRun("r", "r", "ok", True, ()), expected_from="x")  # type: ignore[arg-type]


def test_raw_string_messages_and_parts_without_text() -> None:
    def run(inputs: Any, outputs: Any) -> RunCase | SkippedRun:
        events = (
            {"type": "llm.started", "data": {"input_messages": inputs}},
            {"type": "llm.finished", "data": {"output_messages": outputs}},
        )
        return case_from_run(RecordedRun("k", "k", "ok", True, events))

    made = run("plain question", "plain answer")
    assert isinstance(made, RunCase) and (made.case.input, made.case.expected) == (
        "plain question",
        "plain answer",
    )
    image = [{"role": "user", "parts": [{"type": "image", "content": "x"}]}]
    assert isinstance(run(image, None), SkippedRun)
    assert isinstance(run([{"role": "user", "content": 3}], None), SkippedRun)
    assert isinstance(run("", None), SkippedRun)


def test_write_cases_appends_and_keeps_saved_ids(runs: Path, tmp_path: Path) -> None:
    out = tmp_path / "cases" / "cases.jsonl"
    found = cases_from_runs(runs)
    ids = ["run-bad", "run-good", "run-tool"]
    assert write_cases(out, found.cases) == (3, [])
    written, repeated = write_cases(out, found.cases)
    assert (written, sorted(repeated)) == (0, ids)
    assert sorted(c.id for c in load_cases(out)) == ids
    saved = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    assert {(c["source_run"], c["run_key"]) for c in saved} >= {("good", "good")}


def test_write_cases_refuses_bad_paths_and_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"\.jsonl"):
        write_cases(tmp_path / "cases.csv", [])
    broken = tmp_path / "broken.jsonl"
    broken.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not an eval case"):
        write_cases(broken, [])
    unreadable = tmp_path / "bytes.jsonl"
    unreadable.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ConfigError, match="cannot read"):
        write_cases(unreadable, [])


def test_bad_folders_and_options(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no such runs folder"):
        cases_from_runs(tmp_path / "missing")
    other = tmp_path / "notes.txt"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a run file"):
        cases_from_runs(other)
    with pytest.raises(ConfigError, match="expected_from"):
        cases_from_runs(tmp_path, expected_from="all")  # type: ignore[arg-type]


def test_clipped_and_message_text_edges() -> None:
    from agent_adapter.evals import _run_content as content

    assert content.clipped({"a": {"...": "+3 keys"}}) and content.clipped([["[too deep]"]])
    assert content.clipped(["a", "... +2 items"]) and not content.clipped({"a": [1, "b"]})
    assert content.message_text("hi") == "hi" and content.message_text(3) is None
    assert content.message_text({"content": [{"text": "a"}, {"type": "text", "content": "b"}]})
    assert content.message_text({"parts": "x"}) is None
    assert content.tool_result([{"type": "tool.finished", "call_id": "c2", "data": {}}], "c1") == (
        False,
        None,
    )


def test_a_tool_run_without_a_result_has_no_expected() -> None:
    events = ({"type": "tool.started", "call_id": "t1", "data": {"tool": "x", "arguments": {}}},)
    made = case_from_run(RecordedRun("k", "k", "ok", True, events))
    assert isinstance(made, RunCase) and made.case.input == {} and made.case.expected is None


def test_nested_and_side_calls_never_become_the_question_or_answer() -> None:
    def llm(kind: str, call: str, parent: str = "", **data: Any) -> dict[str, Any]:
        return {"type": kind, "call_id": call, "parent_id": parent, "data": data}

    events = (
        llm("llm.started", "a", stage="tool_loop"),
        llm(
            "llm.finished", "a", stage="tool_loop",
            input_messages=[
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "new question"},
            ],
            output_messages=[{"role": "assistant", "content": "the answer"}],
        ),
        llm("llm.finished", "n", "a", stage="tool_loop", nested=True,
            input_messages=[{"role": "user", "content": "inner"}],
            output_messages=[{"role": "assistant", "content": "inner answer"}]),
        llm("llm.finished", "m", stage="memory",
            input_messages=[{"role": "user", "content": "extract"}],
            output_messages=[{"role": "assistant", "content": "facts"}]),
    )  # fmt: skip
    made = case_from_run(RecordedRun("k", "k", "ok", True, events))
    assert isinstance(made, RunCase)
    assert (made.case.input, made.case.expected) == ("new question", "the answer")
