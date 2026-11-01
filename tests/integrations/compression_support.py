"""Checks every framework compression format must pass: lossless, whole rounds, kept fields."""

from __future__ import annotations

import copy
import json
from typing import Any

from agent_adapter.context import CompressionPolicy, CutLongMessages, MyOwnStep, RemoveOldMessages
from agent_adapter.context.formats import compress_for_sync, convert, detect_format
from agent_adapter.context.formats._shared import Converter
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall

PAD = "x" * 400  # makes an old turn worth removing
DENIED = json.dumps({"error": {"kind": "approval-denied", "message": "no"}})
DROP_OLD = CompressionPolicy(
    start_when_tokens_over=50,
    shrink_to_tokens=10,
    steps=[RemoveOldMessages(keep_first=0, keep_last=1)],
)
CUT = CompressionPolicy(
    start_when_tokens_over=50,
    shrink_to_tokens=10,
    steps=[CutLongMessages(max_tokens_per_message=20)],
)


def summarise_old(messages: list[Message]) -> list[Message]:
    """A step that swaps everything between the first and the last two messages for a summary."""
    summary = Message(role="user", content="summary of earlier turns")
    return [*messages[:1], summary, *messages[-2:]]


SUMMARY = CompressionPolicy(
    start_when_tokens_over=50, shrink_to_tokens=10, steps=[MyOwnStep(summarise_old)]
)


def assert_round_trip(items: list[Any], fmt: str) -> list[Message]:
    """Detected as `fmt`; convert and back gives equal values and the very same objects."""
    snapshot = copy.deepcopy(items)
    assert detect_format(items) == fmt
    conversion = convert(items)
    back = conversion.back(conversion.messages)
    assert back == snapshot
    assert len(back) == len(items) and all(a is b for a, b in zip(back, items, strict=True))
    return conversion.messages


def call_and_result_ids(items: list[Any]) -> tuple[list[str], list[str]]:
    """Call ids and result ids of a native list, read through its converter."""
    messages = convert(items).messages
    calls = [c.id for m in messages for c in m.tool_calls]
    results = [m.tool_call_id for m in messages if m.role == "tool"]
    return calls, results


def assert_drop_keeps_whole_rounds(items: list[Any], *, whole: bool = True) -> list[Any]:
    """Old turns go; every kept call keeps its result; the last item stays.

    With `whole`, every kept item is an input object (no native held a dropped part).
    """
    smaller, report = compress_for_sync(items, DROP_OLD)
    assert report.tokens_after < report.tokens_before and len(smaller) < len(items)
    if whole:
        assert all(any(s is i for i in items) for s in smaller)
    calls, results = call_and_result_ids(smaller) if smaller else ([], [])
    assert sorted(calls) == sorted(results)
    assert smaller[-1] is items[-1]
    return smaller


def assert_cut_keeps_types(items: list[Any]) -> list[Any]:
    """Cutting text keeps the count, the native type of each item and the last item."""
    smaller, report = compress_for_sync(items, CUT)
    assert "cut_long_messages" in report.steps_used
    assert len(smaller) == len(items)
    assert [type(s) for s in smaller] == [type(i) for i in items]
    assert smaller[-1] is items[-1]
    assert call_and_result_ids(smaller) == call_and_result_ids(items)
    return smaller


def assert_summary_is_native(items: list[Any], native_type: type) -> list[Any]:
    """A step's new summary comes back as a native message of the format."""
    smaller, report = compress_for_sync(items, SUMMARY)
    assert report.tokens_after < report.tokens_before
    assert all(isinstance(s, native_type) for s in smaller)
    assert smaller[-1] is items[-1]
    assert "summary of earlier turns" in repr(smaller)
    return smaller


CALL = ToolCall(id="r1", name="lookup", arguments={"q": "a"})
NEW = [
    Message(role="system", content="rules"),
    Message(role="user", content="question"),
    Message(role="assistant", content="", tool_calls=(CALL,)),
    Message(role="tool", tool_call_id="r1", name="lookup", content="result"),
    Message(role="assistant", content="answer"),
]


def render_read_back(converter: type[Converter]) -> list[tuple[Any, ...]]:
    """Natives a format writes for new messages, read back: (role, text, calls, result id)."""
    natives = converter([]).render(list(NEW))
    return [
        (
            m.role,
            m.content,
            [(c.id, c.name, dict(c.arguments)) for c in m.tool_calls],
            m.tool_call_id,
        )
        for m in convert(natives, converter.name).messages
    ]


EXPECTED = [
    ("system", "rules", [], ""),
    ("user", "question", [], ""),
    ("assistant", "", [("r1", "lookup", {"q": "a"})], ""),
    ("tool", "result", [], "r1"),
    ("assistant", "answer", [], ""),
]
