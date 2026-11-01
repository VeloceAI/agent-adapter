import copy
import logging

import pytest

from agent_adapter.cache.providers import AnthropicPromptCache
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.ports import PromptCachePlugin
from agent_adapter.core.types import ModelCaps, ToolCall

ON = ModelCaps(cache_control=True)
CC = {"type": "ephemeral", "ttl": "5m"}
TOOLS = [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {}}]
REQUEST = LLMRequest(
    messages=(
        Message(role="system", content="SYS", cache_breakpoint=True),
        Message(role="user", content="q1"),
        Message(role="assistant", content="", tool_calls=(ToolCall(name="a", id="t1"),)),
        Message(role="tool", content="r1", tool_call_id="t1", name="a"),
        Message(role="user", content="q2", cache_breakpoint=True),
    )
)
KWARGS = {
    "model": "claude-x",
    "max_tokens": 100,
    "system": "SYS",
    "tools": TOOLS,
    "messages": [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "a"}]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
                {"type": "text", "text": "q2"},
            ],
        },
    ],
}


def caps(value: ModelCaps):
    return lambda model: value


async def apply(plugin, kwargs=KWARGS, request=REQUEST):
    before = copy.deepcopy(kwargs)
    out = await plugin.apply(request, kwargs, model="claude-x", client=None)
    assert kwargs == before
    return out


def markers(out) -> int:
    blocks = [*out.get("tools", []), *(out["system"] if isinstance(out["system"], list) else [])]
    for entry in out["messages"]:
        if isinstance(entry["content"], list):
            blocks += entry["content"]
    return sum("cache_control" in b for b in blocks)


def test_plugin_satisfies_the_protocol():
    plugin = AnthropicPromptCache()
    assert isinstance(plugin, PromptCachePlugin)
    assert plugin.providers == frozenset({"anthropic"})
    assert plugin.key_fields() == {"ttl": "5m", "on": ["system", "tools"]}


async def test_gate_off_without_the_capability(caplog):
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.cache.providers.anthropic"):
        default = AnthropicPromptCache()  # default caps: an unknown model has none
        assert await default.apply(REQUEST, KWARGS, model="unknown", client=None) == KWARGS
        assert await apply(AnthropicPromptCache(caps=caps(ModelCaps()))) == KWARGS
    assert "does not take cache_control" in caplog.text


async def test_default_marks_last_system_block_and_last_tool():
    out = await apply(AnthropicPromptCache(caps=caps(ON)))
    assert out["system"] == [{"type": "text", "text": "SYS", "cache_control": CC}]
    assert out["tools"] == [TOOLS[0], {**TOOLS[1], "cache_control": CC}]
    assert out["messages"] == KWARGS["messages"]


async def test_system_block_list_marks_only_the_last_block():
    kwargs = {**KWARGS, "system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}
    out = await apply(AnthropicPromptCache(ttl="1h", on=["system"], caps=caps(ON)), kwargs)
    assert out["system"] == [
        {"type": "text", "text": "A"},
        {"type": "text", "text": "B", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]
    assert out["tools"] == TOOLS


async def test_breakpoints_mark_the_matching_message_blocks():
    request = REQUEST.model_copy(
        update={
            "messages": (
                *REQUEST.messages[:2],
                REQUEST.messages[2].model_copy(update={"cache_breakpoint": True}),
                REQUEST.messages[3].model_copy(update={"cache_breakpoint": True}),
                REQUEST.messages[4],
            )
        }
    )
    out = await apply(AnthropicPromptCache(on=["breakpoints"], caps=caps(ON)), request=request)
    assert out["system"] == "SYS" and out["tools"] == TOOLS
    messages = out["messages"]
    assert messages[0] == {"role": "user", "content": "q1"}
    assert messages[1]["content"][0] == {
        "type": "tool_use",
        "id": "t1",
        "name": "a",
        "cache_control": CC,
    }
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "r1", "cache_control": CC},
        {"type": "text", "text": "q2", "cache_control": CC},
    ]


async def test_string_content_becomes_a_marked_text_block():
    request = LLMRequest(messages=(Message(role="user", content="q1", cache_breakpoint=True),))
    kwargs = {"messages": [{"role": "user", "content": "q1"}]}
    out = await apply(AnthropicPromptCache(on=["breakpoints"], caps=caps(ON)), kwargs, request)
    assert out["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "q1", "cache_control": CC}]}
    ]


async def test_at_most_four_markers_keeping_the_latest(caplog):
    request = LLMRequest(
        messages=tuple(
            Message(role=role, content=f"m{i}", cache_breakpoint=True)
            for i, role in enumerate(["user", "assistant", "user", "assistant", "user"])
        )
    )
    kwargs = {
        "system": "SYS",
        "tools": TOOLS,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
    }
    plugin = AnthropicPromptCache(on=["system", "tools", "breakpoints"], caps=caps(ON))
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.cache.providers.anthropic"):
        out = await apply(plugin, kwargs, request)
    assert markers(out) == 4
    assert [isinstance(e["content"], list) for e in out["messages"]] == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert "cache_control" not in out["tools"][-1]
    assert "3 marker(s) dropped" in caplog.text


async def test_existing_markers_count_towards_the_limit_and_are_not_overwritten():
    host = {"type": "ephemeral"}  # no ttl: Anthropic's default 5m
    kwargs = {
        **KWARGS,
        "tools": [*TOOLS[:1], {**TOOLS[1], "cache_control": host}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "x", "cache_control": host}]}
            for _ in range(3)
        ],
    }
    out = await apply(AnthropicPromptCache(caps=caps(ON)), kwargs)
    assert out["tools"][-1]["cache_control"] == host
    assert markers(out) == 4  # the system marker filled the last free slot
    full = {**kwargs, "system": [{"type": "text", "text": "S", "cache_control": host}]}
    assert await apply(AnthropicPromptCache(caps=caps(ON)), full) == full


async def test_missing_parts_are_skipped():
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "system": ""}
    out = await apply(
        AnthropicPromptCache(on=["system", "tools", "breakpoints"], caps=caps(ON)), kwargs
    )
    assert out == kwargs


@pytest.mark.parametrize("kwargs", [{"ttl": "10m"}, {"on": ["system", "everything"]}])
def test_bad_options_are_rejected(kwargs):
    with pytest.raises(ConfigError):
        AnthropicPromptCache(**kwargs)


async def test_odd_wire_entries_are_tolerated():
    request = LLMRequest(messages=(Message(role="user", content="q", cache_breakpoint=True),))
    kwargs = {
        "messages": [
            "junk",
            {"role": "assistant", "content": None},
            {"role": "user", "content": "q"},
        ]
    }
    out = await apply(AnthropicPromptCache(on=["breakpoints"], caps=caps(ON)), kwargs, request)
    assert out["messages"][:2] == kwargs["messages"][:2]
    assert out["messages"][2]["content"][0]["cache_control"] == CC


async def test_host_markers_with_another_ttl_leave_the_request_unchanged(caplog):
    kwargs = {
        **KWARGS,
        "tools": [*TOOLS[:1], {**TOOLS[1], "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    }
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.cache.providers.anthropic"):
        assert await apply(AnthropicPromptCache(caps=caps(ON)), kwargs) == kwargs
    assert "another ttl" in caplog.text
    out = await apply(AnthropicPromptCache(ttl="1h", caps=caps(ON)), kwargs)
    assert out["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.parametrize("joiner", ["", "\n", "\n\n"])
async def test_assistant_reply_split_into_several_text_blocks_gets_its_breakpoint(joiner):
    request = LLMRequest(
        messages=(
            Message(role="user", content="q"),
            Message(
                role="assistant",
                content=joiner.join(["Part one.", "Part two."]),
                cache_breakpoint=True,
            ),
            Message(role="user", content="next"),
        )
    )
    kwargs = {
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Part one.", "citations": []},
                    {"type": "thinking", "thinking": "...", "signature": "s"},
                    {"type": "text", "text": "Part two."},
                ],
            },
            {"role": "user", "content": "next"},
        ]
    }
    out = await apply(AnthropicPromptCache(on=["breakpoints"], caps=caps(ON)), kwargs, request)
    blocks = out["messages"][1]["content"]
    assert [("cache_control" in b) for b in blocks] == [False, False, True]
    assert out["messages"][2] == {"role": "user", "content": "next"}


async def test_split_assistant_text_must_match_as_a_whole():
    request = LLMRequest(
        messages=(Message(role="assistant", content="Part two.", cache_breakpoint=True),)
    )
    kwargs = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Other."},
                    {"type": "text", "text": "Part three."},
                ],
            }
        ]
    }
    out = await apply(AnthropicPromptCache(on=["breakpoints"], caps=caps(ON)), kwargs, request)
    assert out == kwargs
