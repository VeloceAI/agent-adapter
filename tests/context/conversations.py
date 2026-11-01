"""Conversation builders, a seeded random generator and an independent invariant oracle."""

from __future__ import annotations

import functools
import json
import random
from collections.abc import Sequence

from agent_adapter.context import CompressionPolicy, StepContext, TokenMeter
from agent_adapter.context.conversation import Protection
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall

WORDS = ("alpha", "beta", "gamma", "delta", "tax", "rate", "file", "row", "node", "value")
DENIED = json.dumps({"error": {"kind": "approval-denied", "message": "no"}})


def system(text: str = "You are careful.") -> Message:
    return Message(role="system", content=text)


def user(text: str = "hello", **extra) -> Message:
    return Message(role="user", content=text, **extra)


def assistant(text: str = "ok", **extra) -> Message:
    return Message(role="assistant", content=text, **extra)


def tool_round(
    call_id: str, result: str, *, name: str = "lookup", args: dict | None = None
) -> list[Message]:
    call = ToolCall(name=name, arguments=args or {"q": call_id}, id=call_id)
    return [
        Message(role="assistant", tool_calls=(call,)),
        Message(role="tool", content=result, tool_call_id=call_id, name=name),
    ]


def text(rng: random.Random, words: int) -> str:
    return " ".join(rng.choices(WORDS, k=words))


def context_for(messages: Sequence[Message], target: int, counter=None) -> StepContext:
    """A StepContext over `messages` with `target` tokens, as `compress` would build."""
    return StepContext(target, TokenMeter(counter), Protection(list(messages)))


def tokens(messages: Sequence[Message], counter=None) -> int:
    return TokenMeter(counter).total(messages)


def policy(start: int, target: int, steps=(), **extra) -> CompressionPolicy:
    return CompressionPolicy(
        start_when_tokens_over=start, shrink_to_tokens=target, steps=list(steps), **extra
    )


def random_conversation(rng: random.Random) -> list[Message]:
    """System, user turns, tool rounds (given and synthetic ids), pins, approvals, copies."""
    out: list[Message] = []
    if rng.random() < 0.85:
        out.append(system(text(rng, 6)))
    rounds: list[list[Message]] = []
    for turn in range(rng.randint(1, 10)):
        out.append(user(text(rng, rng.randint(2, 400)), pinned=rng.random() < 0.1))
        for step in range(rng.randint(0, 4)):
            roll = rng.random()
            if roll < 0.15 and rounds:
                out.extend(m.model_copy() for m in rng.choice(rounds))  # a copy of an earlier round
            elif roll < 0.7:
                calls = rng.randint(1, 3)
                given = rng.random() < 0.7
                prefix = f"t{turn}s{step}"
                tool_calls = tuple(
                    ToolCall(name="lookup", arguments={"n": n}, id=f"{prefix}c{n}" if given else "")
                    for n in range(calls)
                )
                round_ = [Message(role="assistant", content=text(rng, 3), tool_calls=tool_calls)]
                for n in range(calls):
                    body = DENIED if rng.random() < 0.08 else text(rng, rng.randint(1, 3000))
                    call_id = f"{prefix}c{n}" if given else f"syn:{prefix}:{n}"
                    round_.append(
                        Message(role="tool", content=body, tool_call_id=call_id, name="lookup")
                    )
                rounds.append(round_)
                out.extend(round_)
            else:
                out.append(assistant(text(rng, rng.randint(1, 800)), pinned=rng.random() < 0.05))
    if rng.random() < 0.3:  # a pending round: some calls not answered yet
        calls = (ToolCall(name="act", id="p1"), ToolCall(name="act", id="p2"))
        out.append(Message(role="assistant", tool_calls=calls))
        if rng.random() < 0.5:
            out.append(Message(role="tool", content="done", tool_call_id="p1", name="act"))
    return out


@functools.cache
def seeded_conversation(seed: int) -> tuple[Message, ...]:
    """`random_conversation` for one seed, built once per test session."""
    return tuple(random_conversation(random.Random(seed)))


# --- the oracle: written apart from the package's own grouping and guard --------------------


def _rounds(messages: Sequence[Message]) -> list[tuple[Message, list[Message]]]:
    found = []
    for i, m in enumerate(messages):
        if m.role == "assistant" and m.tool_calls:
            results = []
            j = i + 1
            while j < len(messages) and messages[j].role == "tool":
                results.append(messages[j])
                j += 1
            found.append((m, results))
    return found


def _tail(messages: Sequence[Message]) -> list[Message]:
    """The last unit: the final message, or the final round with its results."""
    if not messages:
        return []
    end = len(messages)
    start = end - 1
    while start > 0 and messages[start].role == "tool":
        start -= 1
    if not (messages[start].role == "assistant" and messages[start].tool_calls):
        start = end - 1 if messages[-1].role != "tool" else start + 1
    return list(messages[start:end])


def _contains_in_order(haystack: Sequence[Message], needles: Sequence[Message]) -> bool:
    it = iter(haystack)
    return all(any(item == needle for item in it) for needle in needles)


def _comparable(message: Message, system_may_shrink: bool) -> Message:
    """A shortened system message compares equal to its original when shrinking is allowed."""
    if system_may_shrink and message.role == "system" and not message.pinned:
        return message.model_copy(update={"content": ""})
    return message


def system_problems(
    before: Sequence[Message], after: Sequence[Message], *, may_shrink: bool
) -> list[str]:
    """The system messages: all kept, in place, same role and fields; text equal or shorter."""
    old = [m for m in before if m.role == "system"]
    new = [m for m in after if m.role == "system"]
    if len(old) != len(new):
        return ["a system message was added or removed"]
    problems = []
    for a, b in zip(old, new, strict=True):
        if may_shrink and not a.pinned:
            if len(b.content) > len(a.content) or _comparable(a, True) != _comparable(b, True):
                problems.append("a system message changed beyond shorter text")
        elif a != b:
            problems.append("a system message was touched")
    return problems


def invariant_problems(
    before: Sequence[Message], after: Sequence[Message], *, system_may_shrink: bool = False
) -> list[str]:
    """Every broken invariant, as text; [] when `after` is a safe compression of `before`."""
    problems = system_problems(before, after, may_shrink=system_may_shrink)
    end = next((i for i, m in enumerate(before) if m.role != "system"), len(before))
    lead = [_comparable(m, system_may_shrink) for m in before[:end]]
    if [_comparable(m, system_may_shrink) for m in after[: len(lead)]] != lead:
        problems.append("leading system messages changed or moved")
    must_keep = [m for m in before if m.pinned]
    users = [m for m in before if m.role == "user"]
    if users:
        must_keep.append(users[-1])
    must_keep.extend(_tail(before))
    for assistant_msg, results in _rounds(before):
        if any(r.content == DENIED for r in results):
            must_keep.extend([assistant_msg, *results])
    ordered = [m for m in before if any(m is k for k in must_keep)]
    if not _contains_in_order(after, ordered):
        problems.append("a protected message was changed or removed")
    if list(after[-len(_tail(before)) :] if before else []) != _tail(before):
        problems.append("the latest turn is not last and verbatim")

    def shape(round_):
        return round_[0].tool_calls, [r.tool_call_id for r in round_[1]]

    old_rounds = [shape(r) for r in _rounds(before)]
    cursor = 0
    for round_ in _rounds(after):  # each round, in order, is a whole round of the input
        while cursor < len(old_rounds) and old_rounds[cursor] != shape(round_):
            cursor += 1
        if cursor == len(old_rounds):
            problems.append("a tool round is not a whole input round, in order")
            break
        cursor += 1
    for i, m in enumerate(after):
        if m.role == "tool":
            j = i
            while j > 0 and after[j - 1].role == "tool":
                j -= 1
            if j == 0 or not after[j - 1].tool_calls:
                problems.append("an orphan tool result")
    return problems
