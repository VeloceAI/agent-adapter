"""Non-AI steps: RemoveOldMessages, RemoveDuplicates (exact), CutLongMessages, MyOwnStep."""

import pytest

from agent_adapter.context import (
    BaseCompressionStep,
    CompressionStep,
    CutLongMessages,
    MyOwnStep,
    RemoveDuplicates,
    RemoveOldMessages,
)
from agent_adapter.context.conversation import split_units
from agent_adapter.context.steps import shorten, unit_signature
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from tests.context.conversations import assistant, context_for, system, tool_round, user


def chars(messages):
    """One token per character: exact arithmetic in tests."""
    return sum(len(m.content) for m in messages)


def run(step, messages, target):
    return step.apply(list(messages), context_for(messages, target, chars))


# --- RemoveOldMessages -----------------------------------------------------------------------

CHAT = [system("s"), user("task"), assistant("a" * 10), user("b" * 10), assistant("c" * 10)]
CHAT += [user("d" * 10), assistant("e" * 10), user("latest"), assistant("last")]


async def test_remove_old_messages_drops_oldest_first_only_as_needed():
    total = chars(CHAT)
    result = await run(RemoveOldMessages(keep_first=1, keep_last=2), CHAT, total - 15)
    assert result == [m for m in CHAT if m.content not in ("a" * 10, "b" * 10)]


async def test_remove_old_messages_returns_the_input_when_under_target():
    result = await run(RemoveOldMessages(), CHAT, chars(CHAT))
    assert result is not None and result == CHAT


async def test_remove_old_messages_keeps_first_and_last_counts():
    result = await run(RemoveOldMessages(keep_first=2, keep_last=3), CHAT, 0)
    assert [m.content for m in result] == ["s", "task", "a" * 10, "e" * 10, "latest", "last"]


async def test_remove_old_messages_with_zero_keeps_still_spares_protected():
    result = await run(RemoveOldMessages(keep_first=0, keep_last=0), CHAT, 0)
    assert [m.content for m in result] == ["s", "latest", "last"]


async def test_remove_old_messages_keeping_more_than_exists_changes_nothing():
    assert await run(RemoveOldMessages(keep_first=20), CHAT, 0) == CHAT
    assert await run(RemoveOldMessages(keep_last=20), CHAT, 0) == CHAT


async def test_remove_old_messages_removes_a_round_whole_and_keeps_an_edge_round():
    messages = [user("task"), *tool_round("c1", "x" * 50), *tool_round("c2", "y" * 50)]
    messages += [user("latest"), assistant("last")]
    result = await run(RemoveOldMessages(keep_first=1, keep_last=3), messages, 0)
    assert [m.tool_call_id for m in result if m.role == "tool"] == ["c2"]


@pytest.mark.parametrize("settings", [{"keep_first": -1}, {"keep_last": 1.5}, {"keep_first": True}])
def test_remove_old_messages_rejects_bad_counts(settings):
    with pytest.raises(ConfigError, match="int >= 0"):
        RemoveOldMessages(**settings)


# --- RemoveDuplicates ------------------------------------------------------------------------


async def test_remove_duplicates_drops_the_older_copy_of_a_round():
    first = tool_round("c1", "same result" * 5, args={"q": "same"})
    again = tool_round("c2", "same result" * 5, args={"q": "same"})
    messages = [user("task"), *first, assistant("x"), *again, user("latest"), assistant("y")]
    result = await run(RemoveDuplicates(), messages, 0)
    assert result == [messages[0], messages[3], *again, *messages[-2:]]


async def test_remove_duplicates_stops_once_under_target():
    messages = [
        user("dup" * 10),
        user("dup" * 10),
        user("dup" * 10),
        user("latest"),
        assistant("z"),
    ]
    result = await run(RemoveDuplicates(), messages, chars(messages) - 30)
    assert len(result) == 4


async def test_remove_duplicates_keeps_protected_older_copies():
    messages = [user("same", pinned=True), user("same"), user("latest"), assistant("z")]
    assert await run(RemoveDuplicates(), messages, 0) == messages


async def test_remove_duplicates_compares_arguments_not_ids():
    a = tool_round("c1", "r", args={"q": 1})
    b = tool_round("c2", "r", args={"q": 2})
    units = split_units([*a, *b])
    assert unit_signature(units[0]) != unit_signature(units[1])
    c = tool_round("c3", "r", args={"q": 1})
    assert unit_signature(units[0]) == unit_signature(split_units(c)[0])


def test_remove_duplicates_signature_handles_non_json_arguments():
    weird = tool_round("c1", "r", args={"when": object()})
    assert unit_signature(split_units(weird)[0])


def test_remove_duplicates_validates_its_settings():
    with pytest.raises(ConfigError, match="exact"):
        RemoveDuplicates(match="fuzzy")
    with pytest.raises(ConfigError, match="needs a scorer"):
        RemoveDuplicates(match="similar")
    for bad in (1.5, -0.1, True, float("nan"), "0.9"):
        with pytest.raises(ConfigError, match=r"0\.\.1"):
            RemoveDuplicates(same_if_score_above=bad)


# --- CutLongMessages -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        ("start", "abcd\n[... 6 characters cut to save context ...]"),
        ("end", "[... 6 characters cut to save context ...]\nghij"),
        ("start_and_end", "ab\n[... 6 characters cut to save context ...]\nij"),
    ],
)
def test_shorten_keeps_the_chosen_end(keep, expected):
    assert shorten("abcdefghij", 4, keep) == expected


def test_shorten_edges():
    assert shorten("abc", 5, "start") == "abc"
    assert (
        shorten("abcdef", 1, "start_and_end") == "a\n[... 5 characters cut to save context ...]\n"
    )
    assert shorten("abc", 0, "end") == "[... 3 characters cut to save context ...]\n"


async def test_cut_long_messages_cuts_oldest_first_under_the_per_message_cap():
    messages = [user("x" * 1000), assistant("y" * 1000), user("latest"), assistant("last")]
    step = CutLongMessages(max_tokens_per_message=100)
    result = await run(step, messages, chars(messages) - 500)
    assert chars([result[0]]) <= 100 and result[1] is messages[1]
    assert result[0].content.startswith("x" * 50) and "cut to save context" in result[0].content


async def test_cut_long_messages_leaves_protected_and_short_messages():
    messages = [user("short"), user("z" * 500, pinned=True), user("latest" * 100), assistant("q")]
    assert await run(CutLongMessages(max_tokens_per_message=10), messages, 0) == messages


async def test_cut_long_messages_never_touches_tool_calls():
    round_ = tool_round("c1", "r" * 400)
    round_[0] = round_[0].model_copy(update={"content": "t" * 400})
    messages = [user("task"), *round_, user("latest"), assistant("z")]
    result = await run(CutLongMessages(max_tokens_per_message=50), messages, 0)
    assert result[1].tool_calls == round_[0].tool_calls and len(result[1].content) < 100
    assert result[2].tool_call_id == "c1" and len(result[2].content) < 100


async def test_cut_long_messages_refines_for_a_non_linear_counter():
    def heavy(messages):  # the marker costs more than its characters
        return sum(len(m.content) + 40 * m.content.count("[") for m in messages)

    messages = [user("w" * 2000), user("latest"), assistant("z")]
    step = CutLongMessages(max_tokens_per_message=300, keep="end")
    result = await step.apply(list(messages), context_for(messages, 0, heavy))
    assert heavy([result[0]]) <= 300


def test_cut_long_messages_validates_its_settings():
    with pytest.raises(ConfigError, match=">= 1"):
        CutLongMessages(max_tokens_per_message=0)
    with pytest.raises(ConfigError, match="start_and_end"):
        CutLongMessages(keep="middle")


# --- MyOwnStep and the base class --------------------------------------------------------------


async def test_my_own_step_runs_a_sync_function_on_a_copy():
    seen = []

    def drop_first(messages):
        seen.append(messages)
        messages.clear()
        return [Message(role="user", content="replaced")]

    result = await run(MyOwnStep(drop_first), CHAT, 0)
    assert result[0].content == "replaced" and seen[0] is not CHAT and len(CHAT) == 9


async def test_my_own_step_awaits_an_async_function():
    async def keep(messages):
        return messages[:2]

    assert await run(MyOwnStep(function=keep), CHAT, 0) == CHAT[:2]


async def test_my_own_step_rejects_a_bad_return_value():
    with pytest.raises(ConfigError, match="list of Message"):
        await run(MyOwnStep(lambda messages: "nope"), CHAT, 0)
    with pytest.raises(ConfigError, match="needs a function"):
        MyOwnStep("not callable")


def test_steps_satisfy_the_protocol_and_report_their_config():
    steps = [RemoveOldMessages(), RemoveDuplicates(), CutLongMessages(), MyOwnStep(len)]
    assert all(isinstance(step, CompressionStep) for step in steps)
    assert RemoveOldMessages().config() == {
        "step": "remove_old_messages",
        "keep_first": 1,
        "keep_last": 8,
    }
    assert MyOwnStep(len).config()["function"] == "builtins.len"
    assert RemoveDuplicates().config()["scorer"] is None


def test_base_step_config_names_objects_and_lists():
    class Custom(BaseCompressionStep):
        name = "custom"

        async def apply(self, messages, context):
            return messages

    assert Custom().config() == {"step": "custom"}

    from dataclasses import dataclass

    @dataclass
    class Listed(BaseCompressionStep):
        items: tuple = ()
        thing: object = None
        name = "listed"

        async def apply(self, messages, context):
            return messages

    config = Listed(items=(1, "a"), thing=object()).config()
    assert config["items"] == [1, "a"] and config["thing"] == {"type": "builtins.object"}


async def test_cut_long_messages_cuts_the_system_prompt_last_and_only_when_asked():
    messages = [system("s" * 500), user("u" * 500), user("latest"), assistant("z")]
    step = CutLongMessages(max_tokens_per_message=100, include_system_prompt=True)
    result = await run(step, messages, chars(messages) - 300)
    assert result[0] is messages[0] and len(result[1].content) < 200
    result = await run(step, messages, 0)
    assert result[0].role == "system" and len(result[0].content) < 200
    assert await run(CutLongMessages(max_tokens_per_message=100), messages, 0) != messages
    assert (await run(CutLongMessages(max_tokens_per_message=100), messages, 0))[0] is messages[0]


@pytest.mark.parametrize("factory", [CutLongMessages, lambda **kw: MyOwnStep(list, **kw)])
def test_include_system_prompt_must_be_a_bool(factory):
    with pytest.raises(ConfigError, match="must be a bool"):
        factory(include_system_prompt="yes")
