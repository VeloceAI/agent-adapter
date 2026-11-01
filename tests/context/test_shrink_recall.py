"""ShrinkToolResults and recall_tool: shorten old big results, save them, read them back."""

import logging

import pytest

from agent_adapter.context import ShrinkToolResults, recall_tool
from agent_adapter.context.recall import (
    RECALL_TOOL_NAME,
    save_result,
    saved_result_id,
    saved_result_key,
)
from agent_adapter.core.errors import ConfigError, ToolError
from agent_adapter.core.messages import Message
from tests.context.conversations import assistant, context_for, tool_round, user
from tests.fakes.cache import BrokenCache, DictCache


def chars(messages):
    return sum(len(m.content) for m in messages)


def history():
    """Four rounds with big results, then the newest turn."""
    messages = [user("task")]
    for n in range(4):
        messages += tool_round(f"c{n}", f"{n}" * 1000)
    return [*messages, user("latest"), assistant("last")]


async def shrink(messages, target, **settings):
    step = ShrinkToolResults(bigger_than_tokens=500, **settings)
    return await step.apply(list(messages), context_for(messages, target, chars))


async def test_shrink_tool_results_shortens_oldest_first_and_stops_under_target():
    messages = history()
    result = await shrink(messages, chars(messages) - 600)
    assert "shortened to save context" in result[2].content
    assert result[4] is messages[4] and result[6] is messages[6]
    assert result[2].tool_call_id == "c0" and result[2].name == "lookup"
    assert "The full text is not available" in result[2].content
    assert result[2].content.endswith("0" * 200 + "...")


async def test_shrink_tool_results_leaves_the_newest_turns():
    messages = history()
    result = await shrink(messages, 0, older_than_turns=2)
    shortened = [m.tool_call_id for m in result if "shortened" in m.content]
    # Turns: c0..c3 rounds and "last"; the newest two turns are c3 and "last".
    assert shortened == ["c0", "c1", "c2"]
    assert (
        len([m for m in await shrink(messages, 0, older_than_turns=0) if "shortened" in m.content])
        == 4
    )


async def test_shrink_tool_results_skips_small_and_protected_results():
    messages = [user("task"), *tool_round("c1", "x" * 100), *tool_round("c2", "y" * 900)]
    pinned = tool_round("c3", "z" * 900)
    pinned[1] = pinned[1].model_copy(update={"pinned": True})
    messages += [*pinned, assistant("a"), assistant("b"), user("latest"), assistant("last")]
    result = await shrink(messages, 0)
    assert result[2] is messages[2] and result[6] is messages[6]
    assert "shortened" in result[4].content


async def test_shrink_tool_results_returns_the_input_under_target():
    messages = history()
    assert await shrink(messages, chars(messages)) is not None


async def test_shrink_tool_results_saves_and_recall_reads_back():
    cache = DictCache()
    messages = history()
    result = await shrink(messages, 0, save_to=cache)
    result_id = saved_result_id(messages[2])
    assert f'id "{result_id}"' in result[2].content and RECALL_TOOL_NAME in result[2].content
    spec = recall_tool(cache)
    assert spec.name == "recall_context" and "id" in spec.json_schema()["properties"]
    assert await spec.invoke({"id": result_id}) == "0" * 1000


async def test_shrink_tool_results_with_a_broken_store_still_shortens(caplog):
    messages = history()
    with caplog.at_level(logging.WARNING, logger="agent_adapter.context.recall"):
        result = await shrink(messages, chars(messages) - 1000, save_to=BrokenCache())
    assert "The full text is not available" in result[2].content
    assert "could not save tool result" in caplog.text and "cache down" not in caplog.text


def test_shrink_tool_results_validates_its_settings():
    with pytest.raises(ConfigError, match=">= 0"):
        ShrinkToolResults(older_than_turns=-1)
    with pytest.raises(ConfigError, match=">= 1"):
        ShrinkToolResults(bigger_than_tokens=0)
    with pytest.raises(ConfigError, match="CachePort"):
        ShrinkToolResults(save_to="redis://")


def test_saved_result_ids_are_stable_and_content_based():
    a = Message(role="tool", content="x", tool_call_id="1", name="t")
    assert saved_result_id(a) == saved_result_id(a.model_copy())
    assert saved_result_id(a) != saved_result_id(a.model_copy(update={"content": "y"}))
    assert saved_result_id(a).startswith("ctx_") and len(saved_result_id(a)) == 20
    assert saved_result_key(saved_result_id(a)).startswith("aa:ctx:v1:tool_result:global:ctx_")


async def test_save_result_returns_the_id():
    cache = DictCache()
    message = Message(role="tool", content="data", tool_call_id="1", name="t")
    assert await save_result(cache, message) == saved_result_id(message)


@pytest.mark.parametrize(
    "bad", ["../etc", "ctx_123", "CTX_0123456789abcdef", "ctx_0123456789abcdeg"]
)
async def test_recall_rejects_ids_it_did_not_make(bad):
    with pytest.raises(ToolError, match="not a saved result id"):
        await recall_tool(DictCache()).invoke({"id": bad})


async def test_recall_reports_a_missing_or_foreign_entry():
    cache = DictCache()
    spec = recall_tool(cache)
    with pytest.raises(ToolError, match="no saved result"):
        await spec.invoke({"id": "ctx_0123456789abcdef"})
    await cache.set(saved_result_key("ctx_0123456789abcdef"), b"not json")
    with pytest.raises(ToolError, match="no saved result"):
        await spec.invoke({"id": "ctx_0123456789abcdef"})


async def test_recall_with_a_broken_store_raises_a_tool_error(caplog):
    spec = recall_tool(BrokenCache())
    with (
        caplog.at_level(logging.WARNING, logger="agent_adapter.context.recall"),
        pytest.raises(ToolError, match="cannot be read now"),
    ):
        await spec.invoke({"id": "ctx_0123456789abcdef"})
    assert "cache down" not in caplog.text
