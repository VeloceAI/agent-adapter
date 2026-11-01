"""register_format and the ItemsConverter base, on a framework-free fake message type."""

import copy
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from agent_adapter.context import CompressionPolicy, CutLongMessages, MyOwnStep, RemoveOldMessages
from agent_adapter.context.formats import (
    _registry,
    compress_for_sync,
    convert,
    detect_format,
    register_format,
    registered_names,
)
from agent_adapter.context.formats._items import Call, Item, ItemsConverter, Result, Text, as_text
from agent_adapter.context.formats._shared import Present, arguments_of
from agent_adapter.context.formats.openai import OpenAIConverter
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from tests.context.conversations import DENIED, invariant_problems

PAD = "x" * 400


@dataclass(frozen=True)
class FakeMessage:
    """A framework message: a role, items, and a field a converter must keep."""

    role: str
    items: tuple[Any, ...]
    extra: dict[str, Any] = field(default_factory=dict)


class FakeConverter(ItemsConverter):
    """Items are tuples: ("text", s), ("call", id, name, args), ("result", id, name, value)."""

    name = "fake"

    def role_of(self, native: Any) -> str | None:
        return native.role if isinstance(native, FakeMessage) else None

    def items_of(self, native: Any) -> tuple[Any, ...]:
        return native.items

    def read(self, item: Any) -> Item:
        if item[0] == "text":
            return Text(item[1])
        if item[0] == "call":
            return Call(ToolCall(id=item[1], name=item[2], arguments=arguments_of(item[3])))
        if item[0] == "result":
            return Result(item[1], item[2], as_text(item[3]))
        return None

    def tool_result_of(self, native: Any) -> tuple[str, str] | None:
        return (native.extra["call_id"], "") if "call_id" in native.extra else None

    def extra_calls(self, native: Any) -> tuple[ToolCall, ...]:
        return tuple(ToolCall(id=i, name="old") for i in native.extra.get("calls", ()))

    def keep_verbatim(self, native: Any) -> bool:
        return any(item[0] == "approval" for item in native.items)

    def with_items(self, native: Any, items: list[Any]) -> Any:
        return replace(native, items=tuple(items))

    def text_item(self, text: str, like: Any | None) -> Any:
        return ("text", text)

    def with_result(self, item: Any, text: str) -> Any:
        return (*item[:3], text)

    def new(self, message: Message) -> Any:
        return FakeMessage(message.role, (("text", message.content),), {"new": True})


@pytest.fixture(autouse=True)
def fake_format(monkeypatch):
    monkeypatch.setattr(_registry, "_FORMATS", dict(_registry._FORMATS))
    register_format("fake", FakeConverter, detect=lambda item: isinstance(item, FakeMessage))


def conversation() -> list[FakeMessage]:
    return [
        FakeMessage("system", (("text", "Be brief."),)),
        FakeMessage("user", (("text", f"old {PAD}"),), {"id": 1}),
        FakeMessage(
            "assistant", (("text", "look"), ("call", "c1", "a", {}), ("call", "c2", "b", "{}"))
        ),
        FakeMessage(
            "tool", (("result", "c1", "a", {"r": PAD}), ("image",), ("result", "c2", "b", PAD))
        ),
        FakeMessage("assistant", (("call", "c3", "a", None),)),
        FakeMessage("tool", (("result", "c3", "a", DENIED),)),
        FakeMessage("assistant", (("image",), ("text", f"answer {PAD}"), ("text", "more"))),
        FakeMessage("assistant", (("approval",),)),
        FakeMessage("user", (("text", "new"),)),
    ]


def test_round_trip_returns_the_very_same_objects():
    items = conversation()
    snapshot = copy.deepcopy(items)
    conversion = convert(items)
    assert conversion.back(conversion.messages) == snapshot
    assert all(a is b for a, b in zip(conversion.back(conversion.messages), items, strict=True))


def test_each_result_item_is_a_tool_message_with_its_call_id():
    messages = convert(conversation()).messages
    assert [m.role for m in messages] == [
        *("system", "user", "assistant", "tool", "tool", "assistant", "tool"),
        *("assistant", "assistant", "user"),
    ]
    assert [m.tool_call_id for m in messages if m.role == "tool"] == ["c1", "c2", "c3"]
    assert [c.id for c in messages[2].tool_calls] == ["c1", "c2"]
    assert messages[3].content == as_text({"r": PAD})
    assert messages[8].pinned and not messages[7].pinned  # the approval message stays verbatim


@pytest.mark.parametrize(
    "step",
    [
        RemoveOldMessages(keep_first=0, keep_last=1),
        CutLongMessages(max_tokens_per_message=20),
        MyOwnStep(lambda messages: messages[-1:]),
        MyOwnStep(lambda messages: list(reversed(messages))),
    ],
    ids=["drop", "cut", "wipe", "reverse"],
)
def test_compression_keeps_rounds_whole_and_protected_messages(step):
    items = conversation()
    policy = CompressionPolicy(start_when_tokens_over=50, shrink_to_tokens=10, steps=[step])
    smaller, _ = compress_for_sync(items, policy)
    assert invariant_problems(convert(items).messages, convert(smaller).messages) == []
    assert all(isinstance(native, FakeMessage) for native in smaller)


def test_cut_text_keeps_other_items_and_fields():
    items = conversation()
    policy = CompressionPolicy(
        start_when_tokens_over=50,
        shrink_to_tokens=10,
        steps=[CutLongMessages(max_tokens_per_message=20)],
    )
    smaller, _ = compress_for_sync(items, policy)
    assert smaller[1].extra == {"id": 1} and "characters cut" in smaller[1].items[0][1]
    assert smaller[3].items[1] == ("image",) and smaller[3].items[0][:3] == ("result", "c1", "a")
    assert smaller[6].items[0] == ("image",) and len(smaller[6].items) == 2  # one text item left


def test_rebuild_keeps_only_present_results_and_text_goes_first_when_missing():
    items = conversation()
    conversion = convert(items)
    converter = conversion._converter
    group = conversion.groups[3]
    second = group.messages[1].model_copy(update={"content": "short"})
    (rebuilt,) = converter.rebuild(group, [Present(1, second, False)])
    assert rebuilt.items == (("image",), ("result", "c2", "b", "short"))
    assert converter._with_text([("image",)], "t") == [("text", "t"), ("image",)]


def test_whole_native_results_extra_calls_and_bare_tool_messages():
    items = [
        FakeMessage("user", (("text", "q"),)),
        FakeMessage("assistant", (), {"calls": ["k1"]}),
        FakeMessage("tool", (("text", "ok"),), {"call_id": "k1"}),
        FakeMessage("tool", (("text", "loose"),)),
    ]
    messages = convert(items).messages
    assert [c.id for c in messages[1].tool_calls] == ["k1"]
    assert (messages[2].role, messages[2].tool_call_id, messages[2].content) == ("tool", "k1", "ok")
    assert (messages[3].role, messages[3].tool_call_id) == ("tool", "")


def test_summaries_are_rendered_as_natives():
    items = conversation()

    def summarise(messages: list[Message]) -> list[Message]:
        return [messages[0], Message(role="user", content="summary"), *messages[-1:]]

    policy = CompressionPolicy(
        start_when_tokens_over=50, shrink_to_tokens=10, steps=[MyOwnStep(summarise)]
    )
    smaller, _ = compress_for_sync(items, policy)
    assert FakeMessage("user", (("text", "summary"),), {"new": True}) in smaller


def test_a_wrong_item_or_role_is_refused():
    with pytest.raises(ConfigError, match="item 1 is not a fake message"):
        convert([FakeMessage("user", ()), {"role": "user"}], format="fake")
    with pytest.raises(ConfigError, match="not a fake message"):
        convert([FakeMessage("robot", ())], format="fake")


def test_register_format_checks_its_arguments():
    assert "fake" in registered_names()
    with pytest.raises(ConfigError, match="empty or built in"):
        register_format("openai", FakeConverter)
    with pytest.raises(ConfigError, match="empty or built in"):
        register_format("", FakeConverter)
    with pytest.raises(ConfigError, match="Converter subclass"):
        register_format("other", dict)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="must be callable"):
        register_format("other", FakeConverter, detect="yes")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="already registered by"):
        register_format("fake", OpenAIConverter)
    register_format("fake", FakeConverter)  # the same class again: replaced, no error
    assert _registry.registered("fake").detect is None


def test_detection_and_errors_point_at_the_integration_module():
    assert detect_format(conversation()) == "fake"
    register_format("fake", FakeConverter)  # without detect, "auto" no longer finds it
    with pytest.raises(ConfigError, match=r"test_formats_items\.FakeMessage.*integrations module"):
        detect_format(conversation())
    with pytest.raises(ConfigError, match=r"agent_adapter\.integrations\.nope is imported"):
        convert([{"role": "user", "content": "a"}], format="nope")


def test_finish_gets_the_rebuilt_list_last():
    class Reversing(FakeConverter):
        name = "fake_reversed"

        def finish(self, natives: list[Any]) -> list[Any]:
            return natives[::-1]

    register_format("fake_reversed", Reversing)
    items = conversation()[-2:]
    conversion = convert(items, format="fake_reversed")
    assert conversion.back(conversion.messages) == items[::-1]


def test_defaults_and_text_of_values():
    class Plain(FakeConverter):
        tool_result_of = ItemsConverter.tool_result_of
        extra_calls = ItemsConverter.extra_calls
        keep_verbatim = ItemsConverter.keep_verbatim

    native = FakeMessage("assistant", (("approval",),), {"call_id": "k", "calls": ["k"]})
    plain = Plain([native])
    assert plain.tool_result_of(native) is None and plain.extra_calls(native) == ()
    assert not plain.keep_verbatim(native)
    (group,) = plain.groups()
    assert plain.rebuild(group, [Present(0, group.messages[0], True)]) == [native]
    assert (
        as_text(None) == ""
        and as_text("a") == "a"
        and as_text({"b": 1, "a": 2}) == ('{"a": 2, "b": 1}')
    )
