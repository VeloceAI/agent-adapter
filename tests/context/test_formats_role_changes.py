"""A step that changes a message's role or ids gets a new native, never a rebuilt original."""

import pytest

from agent_adapter.context import CompressionPolicy, MyOwnStep
from agent_adapter.context.formats import compress_for_sync, convert
from agent_adapter.core.messages import Message
from tests.context.conversations import invariant_problems
from tests.context.format_cases import CASES


def _as_user(messages: list[Message]) -> list[Message]:
    return [m.model_copy(update={"content": "x", "role": "user"}) for m in messages]


def _new_ids(messages: list[Message]) -> list[Message]:
    return [m.model_copy(update={"tool_call_id": "other"}) for m in messages]


@pytest.mark.parametrize("rewrite", [_as_user, _new_ids], ids=["role", "ids"])
@pytest.mark.parametrize("case", list(CASES))
def test_rewritten_messages_never_become_orphan_results_or_extra_system_prompts(case, rewrite):
    items = CASES[case]()
    policy = CompressionPolicy(
        start_when_tokens_over=50, shrink_to_tokens=10, steps=[MyOwnStep(rewrite)]
    )
    smaller, _ = compress_for_sync(items, policy)
    assert invariant_problems(convert(items).messages, convert(smaller).messages) == []
