import dataclasses
import logging

from agent_adapter.core.context import ToolContext
from agent_adapter.core.types import ApprovalDecision
from tests.fakes.cache import DictCache
from tests.fakes.sinks import RecordingLedger, RecordingSink


def test_first_four_fields_stay_positional():
    names = [f.name for f in dataclasses.fields(ToolContext)]
    assert names == [
        "run_id",
        "call_id",
        "state",
        "extras",
        "thread_id",
        "ledger",
        "cache",
        "sink",
        "approval",
    ]
    ctx = ToolContext("r", "c", {"a": 1}, {"x": 2})
    assert (ctx.run_id, ctx.call_id, ctx.state, ctx.extras) == ("r", "c", {"a": 1}, {"x": 2})
    assert ctx.ledger is None and ctx.cache is None and ctx.sink is None and ctx.approval is None


def test_emit_without_sink_is_a_no_op():
    ToolContext().emit("progress", pct=10)


def test_emit_prefixes_custom_and_carries_ids():
    sink = RecordingSink()
    ctx = ToolContext(run_id="r", call_id="c", sink=sink)
    ctx.emit("progress", pct=10)
    ctx.emit("custom.done")
    assert sink.types() == ["custom.progress", "custom.done"]
    first = sink.events[0]
    assert (first.run_id, first.call_id, dict(first.data)) == ("r", "c", {"pct": 10})


class FailingSink:
    def emit(self, event):
        raise RuntimeError("sink down")


def test_emit_logs_and_survives_a_failing_sink(caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.core.context"):
        ToolContext(run_id="r", sink=FailingSink()).emit("x")
    assert "event sink failed for custom.x" in caplog.text


def test_child_copies_extras_and_drops_approval():
    ledger, cache, sink = RecordingLedger(), DictCache(), RecordingSink()
    parent = ToolContext(
        run_id="r",
        call_id="parent",
        extras={"k": 1},
        thread_id="t",
        ledger=ledger,
        cache=cache,
        sink=sink,
        approval=ApprovalDecision(approved=True),
    )
    child = parent.child(call_id="kid")
    assert child.call_id == "kid" and child.run_id == "r" and child.thread_id == "t"
    assert child.ledger is ledger and child.cache is cache and child.sink is sink
    assert child.approval is None
    child.extras["k"] = 2
    assert parent.extras == {"k": 1}


def test_child_accepts_changes_and_explicit_approval():
    decision = ApprovalDecision(approved=True)
    child = ToolContext(run_id="r").child(call_id="c", run_id="r2", approval=decision)
    assert child.run_id == "r2" and child.approval is decision
    extras = {"only": True}
    assert ToolContext(extras={"a": 1}).child(call_id="c", extras=extras).extras is extras
