"""The shared compression hook pieces: `Shrink`, `wrap_model` and the parts converter base."""

import copy
import runpy
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.context import CompressionPolicy
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from agent_adapter.integrations._compress_hooks import Shrink, wrap_model
from agent_adapter.integrations._compress_parts import PartsConverter, Piece
from tests.integrations.compression_support import DROP_OLD, PAD

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
HISTORY = [{"role": "user", "content": PAD}, {"role": "assistant", "content": PAD}]
HISTORY.append({"role": "user", "content": "new"})
METHODS = {
    "generate": "call",
    "agenerate": "acall",
    "apositional": "acall",
    "stream": "iter",
    "astream": "aiter",
}


class FakeModel:
    """Records the messages each method gets; `name` is state a copy must keep."""

    def __init__(self) -> None:
        self.name = "fake"
        self.seen: list[int] = []

    def generate(self, messages: list[Any], **kwargs: Any) -> str:
        self.seen.append(len(messages))
        return "ok"

    async def agenerate(self, *, messages: list[Any]) -> str:
        self.seen.append(len(messages))
        return "ok"

    async def apositional(self, messages: list[Any] | None = None) -> str:
        self.seen.append(len(messages or ()))
        return "ok"

    def stream(self, messages: list[Any]) -> Any:
        self.seen.append(len(messages))
        yield "chunk"

    async def astream(self, *, messages: list[Any]) -> Any:
        self.seen.append(len(messages))
        yield "chunk"


def shrink() -> Shrink:
    return Shrink(DROP_OLD, "openai")


async def test_wrapped_model_gets_fewer_messages_on_every_kind_of_method() -> None:
    model = FakeModel()
    wrapped = wrap_model(model, shrink(), METHODS)
    assert wrapped.generate(list(HISTORY), temperature=0) == "ok"
    assert wrapped.generate(messages=list(HISTORY)) == "ok"
    assert await wrapped.agenerate(messages=list(HISTORY)) == "ok"
    assert await wrapped.apositional(list(HISTORY)) == "ok"
    assert await wrapped.apositional() == "ok"  # no messages: passed through
    assert list(wrapped.stream(list(HISTORY))) == ["chunk"]
    assert [c async for c in wrapped.astream(messages=list(HISTORY))] == ["chunk"]
    assert wrapped.seen == [1, 1, 1, 1, 0, 1, 1]
    assert type(model) is FakeModel and wrapped is not model  # a shallow copy; same state


def test_wrapped_model_keeps_class_name_state_and_missing_methods() -> None:
    wrapped = wrap_model(FakeModel(), shrink(), {"generate": "call", "absent": "call"})
    assert type(wrapped).__name__ == "FakeModel" and isinstance(wrapped, FakeModel)
    assert wrapped.name == "fake" and not hasattr(wrapped, "absent")
    copied = copy.deepcopy(wrapped)
    copied.generate(list(HISTORY))
    assert copied.seen == [1]


def test_wrapping_again_replaces_the_policy() -> None:
    keep_all = Shrink(
        CompressionPolicy(start_when_tokens_over=10_000, shrink_to_tokens=5), "openai"
    )
    twice = wrap_model(wrap_model(FakeModel(), shrink(), METHODS), keep_all, METHODS)
    twice.generate(list(HISTORY))
    assert twice.seen == [3]
    assert type(twice).__mro__[1] is FakeModel


def test_a_call_without_messages_passes_through() -> None:
    wrapped = wrap_model(FakeModel(), shrink(), METHODS)
    with pytest.raises(TypeError):
        wrapped.generate()


def test_wrap_model_rejects_an_object_with_none_of_the_methods() -> None:
    with pytest.raises(ConfigError, match="not a model"):
        wrap_model(object(), shrink(), METHODS)


@pytest.mark.parametrize(
    ("policy", "pinned", "match"),
    [("nope", None, "CompressionPolicy"), (DROP_OLD, "nope", "pinned must be")],
)
def test_shrink_fails_fast_on_bad_settings(policy: Any, pinned: Any, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        Shrink(policy, "openai", pinned)


def test_shrink_sync_and_pinned() -> None:
    pinned = Shrink(DROP_OLD, "openai", pinned=lambda m: m["content"] == PAD)
    assert len(Shrink(DROP_OLD, "openai").sync(HISTORY)) == 1
    assert pinned.sync(HISTORY)[0] is HISTORY[0]


class Lines(PartsConverter):
    """A toy format: str items are messages, None items ride along."""

    name = "lines"

    def pieces(self, index: int, item: Any) -> list[Piece]:
        return [] if item is None else [(Message(role="user", content=item), None)]

    def retext(self, item: Any, message: Message) -> Any:
        return message.content

    def render(self, messages: list[Message]) -> list[Any]:
        return [m.content for m in messages]


def test_riders_join_the_next_message_else_the_last() -> None:
    groups = Lines([None, "a", None, None, "b", None]).groups()
    assert [g.natives for g in groups] == [(0, 1), (2, 3, 4, 5)]


def test_a_list_of_riders_only_is_rejected() -> None:
    with pytest.raises(ConfigError, match="no item of this lines list"):
        Lines([None, None]).groups()


def test_the_example_runs_the_strands_hook_offline(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLES / "compression_framework_hooks.py"), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == ["strands: 7 -> 3 Be brief."]
