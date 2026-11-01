"""The invariant guard: each rule, on hand-written step outputs."""

import logging

import pytest

from agent_adapter.context import Protection, enforce_invariants
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from tests.context.conversations import assistant, system, tool_round, user

ROUND = tool_round("c1", "result")
BASE = [system(), user("task"), assistant("old"), *ROUND, user("latest"), assistant("last")]


def guard(after, before=BASE):
    return enforce_invariants(before, after, Protection(before), step="test")


def test_guard_passes_a_valid_removal_through():
    after = [m for m in BASE if m.content != "old"]
    assert guard(after) == after


def test_guard_keeps_the_same_objects():
    result = guard(list(BASE))
    assert all(a is b for a, b in zip(result, BASE, strict=True))


def test_guard_maps_equal_copies_back_to_the_input_objects():
    result = guard([m.model_copy() for m in BASE])
    assert all(a is b for a, b in zip(result, BASE, strict=True))


def test_guard_restores_removed_protected_messages(caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.context.guard"):
        result = guard([BASE[2]])
    assert result == [BASE[0], BASE[2], BASE[-2], BASE[-1]]
    assert "skipped 3 change(s)" in caplog.text and "latest" not in caplog.text


def test_guard_reverts_a_change_to_a_protected_message():
    after = list(BASE)
    after[0] = system("new rules")
    after[-1] = assistant("rewritten")
    assert guard(after) == BASE


def test_guard_allows_an_in_place_change_to_an_unprotected_message():
    after = list(BASE)
    after[4] = after[4].model_copy(update={"content": "short"})
    assert guard(after)[4].content == "short"


def test_guard_keeps_a_tool_round_whole():
    without_result = [m for m in BASE if m is not ROUND[1]]
    assert guard(without_result) == BASE
    without_call = [m for m in BASE if m is not ROUND[0]]
    assert guard(without_call) == BASE


def test_guard_allows_removing_a_whole_round():
    after = [m for m in BASE if m not in ROUND]
    assert guard(after) == after


def test_guard_drops_added_system_and_tool_messages_and_calls():
    call = ToolCall(name="x", id="new")
    added = [
        system("sneaky"),
        Message(role="tool", content="r", tool_call_id="new"),
        Message(role="assistant", tool_calls=(call,)),
    ]
    after = [*BASE[:2], *added, *BASE[2:]]
    assert guard(after) == BASE


def test_guard_rejects_a_changed_tool_call_in_place():
    after = list(BASE)
    after[3] = ROUND[0].model_copy(update={"tool_calls": (ToolCall(name="other", id="c1"),)})
    assert guard(after) == BASE


def test_guard_places_an_added_message_after_the_system_lead():
    summary = user("summary of earlier turns")
    result = guard([summary, *BASE])
    assert result[0] is BASE[0] and result[1] is summary


def test_guard_places_an_added_message_before_the_latest_turn():
    note = user("note")
    result = guard([*BASE, note])
    assert result[-3:] == [note, BASE[-2], BASE[-1]] and result[-2] is BASE[-2]


def test_guard_never_puts_an_added_message_between_the_latest_user_and_a_pending_round():
    pending = tool_round("p1", "")[:1]  # the call, no result yet
    before = [system(), user("task"), assistant("old"), user("latest"), *pending]
    note = user("note")
    result = guard([*before[:4], note, *pending], before=before)
    assert result == [*before[:3], note, before[3], *pending]


def test_guard_moves_an_added_message_out_of_a_tool_round():
    note = user("note")
    after = [*BASE[:4], note, *BASE[4:]]
    result = guard(after)
    assert result.index(note) == result.index(ROUND[1]) + 1


def test_guard_replaces_a_span_with_a_summary():
    summary = user("summary")
    after = [BASE[0], BASE[1], summary, BASE[-2], BASE[-1]]
    assert guard(after) == after


def test_guard_keeps_a_moved_tool_round_in_place():
    after = [BASE[0], *ROUND, BASE[1], BASE[2], BASE[-2], BASE[-1]]
    assert guard(after) == BASE


def test_guard_keeps_a_moved_protected_message_in_place_without_a_copy():
    after = [BASE[0], BASE[-2], *BASE[1:-2], BASE[-1]]
    assert guard(after) == BASE


def test_guard_allows_a_step_to_repeat_a_plain_message_but_not_a_tool_result():
    assert guard([*BASE[:2], BASE[1], *BASE[2:]]) == [*BASE[:2], BASE[1], *BASE[2:]]
    assert guard([*BASE[:5], ROUND[1], *BASE[5:]]) == BASE


def test_guard_allows_moving_an_unprotected_plain_message():
    old = BASE[2]
    after = [BASE[0], BASE[1], *ROUND, old, BASE[-2], BASE[-1]]
    assert guard(after) == after


def test_guard_keeps_a_repeated_object_in_its_slots():
    shared = assistant("again")
    before = [user("a"), shared, user("b"), shared, user("c"), assistant("d")]
    assert guard(list(before), before) == before
    result = guard([before[0], before[2], shared, *before[4:]], before)
    assert result == [before[0], before[2], shared, *before[4:]]


def test_guard_with_an_empty_input_accepts_plain_messages():
    note = user("hi")
    assert enforce_invariants([], [note], Protection([]), step="test") == [note]


@pytest.mark.parametrize("bad", [None, "text", [1, 2], ({"role": "user"},)])
def test_guard_rejects_output_that_is_not_messages(bad):
    with pytest.raises(ConfigError, match="list of Message"):
        guard(bad)


def shorten_guard(after, before=BASE, allow=True):
    return enforce_invariants(
        before, after, Protection(before), step="test", allow_system_prompt_shortening=allow
    )


def test_guard_allows_only_a_shorter_system_text_when_asked():
    short = BASE[0].model_copy(update={"content": "You are."})
    assert shorten_guard([short, *BASE[1:]])[0] is short
    assert shorten_guard([short, *BASE[1:]], allow=False) == BASE
    longer = BASE[0].model_copy(update={"content": BASE[0].content + "!"})
    flagged = BASE[0].model_copy(update={"content": "You.", "cache_breakpoint": True})
    for bad in (longer, flagged):
        assert shorten_guard([bad, *BASE[1:]]) == BASE
    as_user = Message(role="user", content="You.")  # a new message; the system one stays
    assert shorten_guard([as_user, *BASE[1:]]) == [BASE[0], as_user, *BASE[1:]]


def test_guard_keeps_a_shortened_system_slot_when_neighbours_change():
    short = BASE[0].model_copy(update={"content": "You."})
    result = shorten_guard([short, BASE[-2], BASE[-1]])
    assert result == [short, BASE[-2], BASE[-1]]
    longer = BASE[0].model_copy(update={"content": BASE[0].content + " and more"})
    assert shorten_guard([longer, BASE[-2], BASE[-1]]) == [BASE[0], BASE[-2], BASE[-1]]


def test_guard_never_lets_a_pinned_system_message_shrink():
    pinned = system("Rules.").model_copy(update={"pinned": True})
    before = [pinned, user("q"), assistant("a")]
    short = pinned.model_copy(update={"content": "R."})
    assert shorten_guard([short, *before[1:]], before) == before
