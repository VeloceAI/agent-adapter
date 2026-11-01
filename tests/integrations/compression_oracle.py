"""Format checks against the independent invariant oracle, for every compression step."""

import copy
import runpy
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from agent_adapter.context import (
    CompressionPolicy,
    CompressionStep,
    CutLongMessages,
    MyOwnStep,
    RemoveOldMessages,
    ShrinkToolResults,
)
from agent_adapter.context.formats import Pinned, compress_for_sync, convert, detect_format
from agent_adapter.core.messages import Message
from tests.context.conversations import DENIED, invariant_problems
from tests.context.test_invariants_property import ADVERSARIAL

PAD = "x" * 400
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
__all__ = ["DENIED", "PAD", "STEPS", "check_round_trip", "check_safe", "example", "policy"]


def summarise(messages: list[Message]) -> list[Message]:
    """A step function that puts one summary where the old turns were."""
    return [messages[0], Message(role="user", content="summary of old turns"), *messages[-1:]]


STEPS: dict[str, Callable[[], CompressionStep]] = {
    "drop_old": lambda: RemoveOldMessages(keep_first=0, keep_last=1),
    "cut": lambda: CutLongMessages(max_tokens_per_message=20),
    "shrink": lambda: ShrinkToolResults(bigger_than_tokens=20, older_than_turns=0),
    "summary": lambda: MyOwnStep(summarise),
    **{f"own_{name}": (lambda fn=fn: MyOwnStep(fn)) for name, fn in ADVERSARIAL.items()},
}


def policy(step: CompressionStep) -> CompressionPolicy:
    """A policy that always runs `step` and aims far below any test conversation."""
    return CompressionPolicy(start_when_tokens_over=50, shrink_to_tokens=10, steps=[step])


def check_round_trip(
    items: Sequence[Any], format: str, dump: Callable[[Any], Any] = lambda item: item
) -> None:
    """Detected as `format`; converting and back gives equal values and the same objects."""
    snapshot = [dump(item) for item in copy.deepcopy(list(items))]
    assert detect_format(items) == format
    conversion = convert(items)
    back = conversion.back(conversion.messages)
    assert [dump(item) for item in back] == snapshot
    assert all(a is b for a, b in zip(back, items, strict=True))


def check_safe(items: Sequence[Any], step: CompressionStep, pinned: Pinned | None = None) -> list:
    """Compress `items` with one step; no orphan or partial tool round, protected kept."""
    before = convert(items, pinned=pinned).messages
    smaller, _ = compress_for_sync(items, policy(step), pinned=pinned)
    after = convert(smaller, pinned=pinned).messages
    assert invariant_problems(before, after) == []
    assert all(type(native) in {type(item) for item in items} for native in smaller)
    return smaller


def example(name: str) -> dict[str, Any]:
    """The globals of `examples/<name>.py`."""
    return runpy.run_path(str(EXAMPLES / f"{name}.py"))
