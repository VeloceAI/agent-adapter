"""`reducer(rule)`: merge rules as LangGraph channels that see every write of a superstep."""

import operator
from typing import Annotated, Any, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.errors import EmptyChannelError
from langgraph.graph import END, START, StateGraph

from agent_adapter.core.errors import FlowError, ToolUsageError
from agent_adapter.flow import Merge, keyed
from agent_adapter.integrations.langgraph import reducer
from agent_adapter.integrations.langgraph._channels import COUNTS, WRITE_KEY, merge_channel, tagged

pytestmark = pytest.mark.contract


class Board(TypedDict, total=False):
    note: Annotated[str, reducer(Merge.replace)]
    log: Annotated[list[str], reducer(Merge.append)]
    total: Annotated[int, reducer(Merge.sum)]
    hits: Annotated[list[dict[str, Any]], reducer(keyed(by="id"))]
    plain: Annotated[list[str], operator.add]


def fan(writes: dict[str, dict[str, Any]]) -> Any:
    """START -> start -> every node in `writes` (one superstep) -> END."""
    graph = StateGraph(Board)
    graph.add_node("start", lambda state: {"log": ["start"]})
    graph.add_edge(START, "start")
    for name, update in writes.items():
        graph.add_node(name, lambda state, update=update: update)
        graph.add_edge("start", name)
        graph.add_edge(name, END)
    return graph.compile()


async def test_every_rule_merges_the_writes_of_one_superstep():
    app = fan(
        {
            "b": {"note": "same", "log": ["b"], "total": 2, "hits": [{"id": 1, "v": "b"}]},
            "a": {"note": "same", "log": ["a1", "a2"], "total": 3, "hits": [{"id": 1, "v": "a"}]},
        }
    )
    out = await app.ainvoke({"note": "", "log": [], "total": 1, "hits": [{"id": 0, "v": "x"}]})
    assert out["note"] == "same"  # equal replace writes are not a conflict
    assert out["log"] == ["start", "a1", "a2", "b"]  # task (node-name) order, items in order
    assert out["total"] == 6
    assert out["hits"] == [{"id": 0, "v": "x"}, {"id": 1, "v": "b"}]  # "b" sorts last: it wins


async def test_different_replace_writes_in_one_superstep_conflict():
    app = fan({"a": {"note": "left"}, "b": {"note": "right"}})
    with pytest.raises(FlowError, match="conflicting writes to note"):
        await app.ainvoke({"note": ""})


async def test_a_value_the_rule_cannot_merge_is_a_flow_error():
    app = fan({"a": {"total": "x"}})
    with pytest.raises(FlowError, match="cannot merge total with rule sum: TypeError"):
        await app.ainvoke({"total": 1})
    keyless = fan({"a": {"hits": [{"other": 1}]}})
    with pytest.raises(FlowError, match="item without key 'id' written to hits"):
        await keyless.ainvoke({"hits": []})


def test_reducer_needs_a_merge_rule():
    with pytest.raises(ToolUsageError, match="MergeRule"):
        reducer("append")  # type: ignore[arg-type]


def _channel(rule: Any = Merge.append) -> Any:
    channel = merge_channel(rule)
    channel.key = "log"
    return channel


def test_tagged_writes_merge_by_phase_then_node_then_sequence():
    channel = _channel()
    assert channel.update([["init"]]) is True  # a raw write to an empty channel sets it
    writes = [
        tagged("zeta", 1, 0, ["z"]),
        tagged("alpha", 1, 1, ["a2"]),
        tagged("alpha", 1, 0, ["a1"]),
        tagged("gate", 0, 0, ["edit"]),
    ]
    channel.update(writes)
    assert channel.get() == ["init", "edit", "a1", "a2", "z"]
    assert channel.update([]) is False


def test_each_edit_merges_on_its_own_before_node_writes():
    channel = _channel(Merge.replace)
    channel.update(["start"])
    channel.update([tagged("b", 0, 0, "edit-b"), tagged("a", 0, 0, "edit-a")])
    assert channel.get() == "edit-b"  # edits apply one after another, in node order
    channel.update([tagged("a", 0, 0, "e"), tagged("a", 1, 0, "n")])
    assert channel.get() == "n"


def test_raw_writes_to_a_set_channel_merge_by_the_rule():
    channel = _channel()
    channel.update([["a"]])
    channel.update([["b"], ["c"]])
    assert channel.get() == ["a", "b", "c"]


def test_channels_copy_checkpoint_and_restore():
    channel = _channel()
    empty = channel.copy()
    with pytest.raises(EmptyChannelError):
        empty.get()
    assert not empty.is_available()
    channel.update([["a"]])
    restored = channel.from_checkpoint(channel.checkpoint())
    assert restored.get() == ["a"] and restored.key == "log" and restored.is_available()
    assert channel.copy().get() == ["a"]
    assert not channel.from_checkpoint(empty.checkpoint()).is_available()
    assert channel == _channel() and channel != merge_channel(Merge.sum)
    assert hash(channel) == hash(_channel())
    assert channel.ValueType is Any and channel.UpdateType is Any


def test_counts_merge_dicts_key_by_key():
    channel = _channel(COUNTS)
    channel.update([{}])
    channel.update([tagged("b", 1, 0, {"b": 1}), tagged("a", 1, 0, {"a": 2, "b": 0})])
    assert channel.get() == {"a": 2, "b": 1}


@pytest.mark.parametrize(
    "write",
    [
        {WRITE_KEY: ["a", 1, 0]},  # no value
        {WRITE_KEY: ["a", 1, 0], "value": 1, "extra": 2},
        {WRITE_KEY: ["a", 7, 0], "value": 1},  # unknown phase
        {WRITE_KEY: ["a", 1], "value": 1},
        {WRITE_KEY: [1, 1, 0], "value": 1},
        {WRITE_KEY: ["a", True, 0], "value": 1},
        {WRITE_KEY: "a", "value": 1},
    ],
)
def test_a_malformed_tagged_write_is_a_flow_error(write):
    with pytest.raises(FlowError, match="malformed tagged write to log"):
        _channel().update([write])


def test_user_values_that_look_like_the_old_marker_are_plain():
    channel = _channel()
    channel.update([[{"__aa_write__": ["x", 1, 0], "value": 1}]])
    assert channel.get() == [{"__aa_write__": ["x", 1, 0], "value": 1}]
