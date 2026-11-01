import dataclasses

import pytest

from agent_adapter.cache import digest
from agent_adapter.cache.layout import PromptLayout
from agent_adapter.core.messages import Message

HISTORY = (Message(role="user", content="q1"), Message(role="assistant", content="a1"))


def test_system_text_is_static_then_sections_in_order():
    layout = PromptLayout("You are terse.", sections={"tools": "T", "examples": "E", "empty": ""})
    assert layout.system_text() == "You are terse.\n\nT\n\nE"


def test_messages_put_the_static_prefix_first_and_dynamic_text_last():
    layout = PromptLayout("SYS", dynamic="Today: 2026-10-29")
    messages = layout.messages("What changed?", history=HISTORY)
    assert messages == (
        Message(role="system", content="SYS", cache_breakpoint=True),
        *HISTORY,
        Message(role="user", content="Today: 2026-10-29\n\nWhat changed?"),
    )


@pytest.mark.parametrize(
    ("dynamic", "user", "last"),
    [("", "hi", "hi"), ("ctx", None, "ctx"), ("ctx", "", "ctx")],
)
def test_user_turn_joins_only_non_empty_parts(dynamic, user, last):
    messages = PromptLayout("S", dynamic=dynamic).messages(user)
    assert messages[-1] == Message(role="user", content=last)


def test_no_user_turn_without_dynamic_or_user_text_and_no_empty_system():
    assert PromptLayout("S").messages() == (
        Message(role="system", content="S", cache_breakpoint=True),
    )
    assert PromptLayout("").messages("hi", history=HISTORY) == (
        *HISTORY,
        Message(role="user", content="hi"),
    )


def test_prefix_digest_tracks_only_the_static_part():
    base = PromptLayout("S", dynamic="one")
    assert base.prefix_digest() == digest("S")
    assert base.prefix_digest() == PromptLayout("S", dynamic="two").prefix_digest()
    assert base.prefix_digest() != base.with_section("x", "y").prefix_digest()


def test_with_helpers_return_new_layouts():
    base = PromptLayout("S", sections={"a": "1", "b": "2"})
    replaced = base.with_section("a", "1b")
    added = base.with_section("c", "3")
    assert replaced.system_text() == "S\n\n1b\n\n2"
    assert added.system_text() == "S\n\n1\n\n2\n\n3"
    assert base.with_static("T").system_text() == "T\n\n1\n\n2"
    assert base.system_text() == "S\n\n1\n\n2"


def test_layout_is_frozen_hashable_and_copies_sections():
    sections = {"a": "1"}
    layout = PromptLayout("S", sections=sections)
    sections["a"] = "changed"
    assert layout.system_text() == "S\n\n1"
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.static = "x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        layout.sections["a"] = "x"  # type: ignore[index]
    assert hash(layout) == hash(PromptLayout("S", sections={"a": "other"}))
    assert layout == PromptLayout("S", sections={"a": "1"})
