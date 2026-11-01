"""Units and protection: rounds are atomic; system, pins, approvals and the newest turn are safe."""

import json

from agent_adapter.context import Protection, carries_approval_state, split_units
from agent_adapter.core.messages import Message
from tests.context.conversations import DENIED, assistant, system, tool_round, user


def test_split_units_groups_a_tool_round_with_its_results():
    messages = [system(), user(), *tool_round("c1", "r"), assistant("done"), user("next")]
    units = split_units(messages)
    assert [(u.kind, u.start, u.stop) for u in units] == [
        ("system", 0, 1),
        ("user", 1, 2),
        ("tool_round", 2, 4),
        ("assistant", 4, 5),
        ("user", 5, 6),
    ]
    assert units[2].messages == tuple(messages[2:4])


def test_split_units_marks_results_without_a_call():
    messages = [user(), Message(role="tool", content="x"), Message(role="tool", content="y")]
    assert [(u.kind, u.stop) for u in split_units(messages)] == [("user", 1), ("tool_results", 3)]


def test_protection_covers_system_pins_latest_user_and_last_unit():
    messages = [
        system(),
        user("first"),
        assistant("a"),
        user("pinned", pinned=True),
        assistant("b"),
        user("latest"),
        *tool_round("c1", "r"),
    ]
    protection = Protection(messages)
    flags = [protection.is_protected(m) for m in messages]
    assert flags == [True, False, False, True, False, True, True, True]


def test_protection_covers_a_whole_round_with_approval_state():
    denied = tool_round("c1", DENIED)
    messages = [user("a"), *denied, assistant("b"), user("c"), assistant("d")]
    protection = Protection(messages)
    assert protection.is_protected(denied[0]) and protection.is_protected(denied[1])
    assert not protection.is_protected(messages[3])


def test_protection_is_by_object_and_pins_are_intrinsic():
    messages = [user("latest"), assistant("last")]
    protection = Protection(messages)
    assert not protection.is_protected(messages[0].model_copy())
    assert protection.is_protected(Message(role="user", pinned=True))
    assert protection.is_protected(Message(role="system"))


def test_protection_of_an_empty_conversation_is_empty():
    assert not Protection([]).is_protected(user())
    assert split_units([]) == []


def test_split_units_flags_protected_units():
    messages = [user("a"), assistant("b", pinned=True), assistant("c"), user("d"), assistant("e")]
    assert [u.protected for u in split_units(messages)] == [False, True, False, True, True]


def test_carries_approval_state_reads_the_tool_error_kind():
    def tool(content):
        return Message(role="tool", content=content)

    for kind in ("approval-denied", "approval-pending", "guardrail-blocked", "budget-exceeded"):
        assert carries_approval_state(tool(json.dumps({"error": {"kind": kind, "message": ""}})))
    assert not carries_approval_state(tool(json.dumps({"error": {"kind": "tool-failed"}})))
    assert not carries_approval_state(tool('{"error": not json'))
    assert not carries_approval_state(tool('{"error": "text"}'))
    assert not carries_approval_state(tool("plain"))
    assert not carries_approval_state(Message(role="user", content=DENIED))
