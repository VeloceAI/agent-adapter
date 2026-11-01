"""Approval state in every format is protected: no step, in any order, ever drops it."""

import itertools

import pytest

from agent_adapter.context import (
    CompressionPolicy,
    CutLongMessages,
    MyOwnStep,
    RemoveDuplicates,
    RemoveOldMessages,
    ShrinkToolResults,
    Summarize,
    SummarizeLongMessage,
    carries_approval_state,
    compress,
    has_approval_marker,
)
from agent_adapter.context.formats import compress_for_sync, convert
from agent_adapter.core.messages import Message, ProviderRaw
from tests.context.ai_fakes import FakeSummariser
from tests.context.format_cases import APPROVAL_CASES, DENIED, FORMAT_OF, PAD


def _keep_last(messages):
    return messages[-1:]


def _steps():
    return {
        "remove_old": RemoveOldMessages(keep_first=0, keep_last=1),
        "cut": CutLongMessages(max_tokens_per_message=20),
        "shrink": ShrinkToolResults(older_than_turns=0, bigger_than_tokens=10),
        "dedupe": RemoveDuplicates(),
        "summarize": Summarize(FakeSummariser(), keep_recent_messages=0, summary_max_tokens=20),
        "long": SummarizeLongMessage(FakeSummariser(), bigger_than_tokens=10),
        "drop_all": MyOwnStep(_keep_last),
    }


def _policy(steps):
    return CompressionPolicy(start_when_tokens_over=20, shrink_to_tokens=1, steps=list(steps))


def _approval_items(items):
    return [item for item in items if has_approval_marker(item) or DENIED in str(item)]


ORDERS = [order for size in (1, 2, 3) for order in itertools.permutations(_steps(), size)]


@pytest.mark.parametrize("case", list(APPROVAL_CASES))
def test_no_step_in_any_order_drops_an_approval_item(case):
    for order in ORDERS:
        items = APPROVAL_CASES[case]()
        marked = _approval_items(items)
        assert marked, case
        steps = _steps()
        smaller, _ = compress_for_sync(
            items, _policy(steps[name] for name in order), format=FORMAT_OF.get(case, case)
        )
        for item in marked:
            assert any(out is item for out in smaller), (case, order, item)


@pytest.mark.parametrize("case", list(APPROVAL_CASES))
def test_approval_items_become_protected_messages(case):
    conversion = convert(APPROVAL_CASES[case](), FORMAT_OF.get(case, case))
    protected = [m for m in conversion.messages if m.pinned or carries_approval_state(m)]
    assert protected, case


def test_the_mcp_approval_repro_keeps_both_items():
    items = APPROVAL_CASES["openai_responses"]()
    policy = CompressionPolicy(
        start_when_tokens_over=100,
        shrink_to_tokens=50,
        steps=[RemoveOldMessages(keep_first=0, keep_last=1)],
    )
    smaller, _ = compress_for_sync(items, policy)
    kinds = [item.get("type") for item in smaller]
    assert "mcp_approval_request" in kinds and "mcp_approval_response" in kinds


async def test_our_own_messages_with_approval_raw_items_are_protected():
    raw = ProviderRaw(
        provider="openai",
        api="responses",
        items=[{"type": "mcp_approval_request", "id": "mcpr_1", "name": "t", "arguments": "{}"}],
    )
    approval = Message(role="assistant", content="", raw=raw)
    messages = [
        Message(role="user", content=f"old {PAD}"),
        approval,
        Message(role="assistant", content=f"answer {PAD}"),
        Message(role="user", content="new"),
    ]
    smaller, _ = await compress(messages, _policy([RemoveOldMessages(keep_first=0, keep_last=1)]))
    assert carries_approval_state(approval) and any(m is approval for m in smaller)


@pytest.mark.parametrize(
    "native",
    [
        {"type": "mcp_approval_request"},
        {"type": "mcp_approval_response", "approve": True},
        {"approval_request_id": "x"},
        {"content": [{"type": "tool_approval", "text": "ok"}]},
        {"parts": [{"function_call": {"name": "adk_request_confirmation"}}]},
        {"parts": [{"functionResponse": {"name": "request_approval", "response": {}}}]},
    ],
)
def test_approval_markers_are_found_anywhere_in_a_native(native):
    assert has_approval_marker(native)


@pytest.mark.parametrize(
    "native",
    [
        {"role": "user", "content": "please approve my leave"},
        {"type": "function_call", "name": "approve_refund"},
        {"type": "message", "content": [{"type": "output_text", "text": "approval needed"}]},
        {"parts": [{"functionCall": {"name": "confirm_order_total"}}]},
        "approval",
        None,
    ],
)
def test_ordinary_natives_carry_no_approval_marker(native):
    assert not has_approval_marker(native)


def test_the_marker_walk_is_bounded_and_survives_cycles():
    loop: dict = {"role": "user"}
    loop["self"] = loop
    deep: dict = {"type": "message"}
    for _ in range(50):
        deep = {"inner": deep}
    deep_marker = {"type": "mcp_approval_request"}
    for _ in range(50):
        deep_marker = {"inner": deep_marker}
    assert not has_approval_marker(loop) and not has_approval_marker(deep)
    assert not has_approval_marker(deep_marker)  # too deep to be a message's own field

    class Item:
        def __init__(self):
            self.type = "mcp_approval_response"

    assert has_approval_marker([Item()])
