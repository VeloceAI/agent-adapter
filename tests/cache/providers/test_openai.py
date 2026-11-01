import copy
import logging

import pytest

from agent_adapter.cache import digest
from agent_adapter.cache.providers import OpenAIPromptCache, OpenAIPromptCacheV2
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.ports import PromptCachePlugin
from agent_adapter.core.types import ModelCaps

SYSTEM = Message(role="system", content="You are terse.", cache_breakpoint=True)
REQUEST = LLMRequest(messages=(SYSTEM, Message(role="user", content="hi")))
KWARGS = {"model": "gpt-4.1", "messages": [{"role": "system", "content": "You are terse."}]}

V1 = ModelCaps(prompt_cache_key=True)
V1_24H = ModelCaps(prompt_cache_key=True, prompt_cache_retention_24h=True)
V2 = ModelCaps(prompt_cache_options=True)


def lookup(caps: ModelCaps):
    seen = []

    def caps_of(model: str) -> ModelCaps:
        seen.append(model)
        return caps

    caps_of.seen = seen
    return caps_of


async def apply(plugin, kwargs=KWARGS, request=REQUEST, model="gpt-4.1"):
    before = copy.deepcopy(kwargs)
    out = await plugin.apply(request, kwargs, model=model, client=None)
    assert kwargs == before  # never edited in place
    return out


def test_plugins_satisfy_the_protocol():
    for plugin in (OpenAIPromptCache(), OpenAIPromptCacheV2()):
        assert isinstance(plugin, PromptCachePlugin)
        assert plugin.providers == frozenset({"openai"})


async def test_v1_gate_off_for_unsupported_models(caplog):
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.cache.providers.openai"):
        assert await apply(OpenAIPromptCache(retention="in_memory"), model="unknown") == KWARGS
        assert await apply(OpenAIPromptCache(caps=lookup(ModelCaps()))) == KWARGS
    assert "does not take prompt_cache_key" in caplog.text


async def test_v1_default_key_is_the_system_digest_and_caps_get_the_model():
    caps = lookup(V1)
    out = await apply(OpenAIPromptCache(caps=caps), model="gpt-4.1-mini")
    assert out == {**KWARGS, "prompt_cache_key": digest("You are terse.")}
    assert caps.seen == ["gpt-4.1-mini"]


async def test_v1_explicit_and_callable_keys():
    out = await apply(OpenAIPromptCache(key="tenant-42", caps=lookup(V1)))
    assert out["prompt_cache_key"] == "tenant-42"
    out = await apply(OpenAIPromptCache(key=lambda r: f"n{len(r.messages)}", caps=lookup(V1)))
    assert out["prompt_cache_key"] == "n2"


async def test_v1_no_key_without_a_system_message():
    request = LLMRequest(messages=(Message(role="user", content="hi"),))
    out = await apply(OpenAIPromptCache(caps=lookup(V1)), request=request)
    assert "prompt_cache_key" not in out


@pytest.mark.parametrize(
    ("retention", "caps", "expected"),
    [
        ("in_memory", V1, "in_memory"),
        ("24h", V1_24H, "24h"),
        ("24h", V1, None),
        (None, V1_24H, None),
    ],
)
async def test_v1_retention_gate(retention, caps, expected):
    out = await apply(OpenAIPromptCache(retention=retention, caps=lookup(caps)))
    assert out.get("prompt_cache_retention") == expected
    assert "prompt_cache_key" in out


async def test_v1_keeps_values_already_in_kwargs():
    kwargs = {**KWARGS, "prompt_cache_key": "host", "prompt_cache_retention": "in_memory"}
    out = await apply(OpenAIPromptCache(retention="24h", caps=lookup(V1_24H)), kwargs=kwargs)
    assert out["prompt_cache_key"] == "host"
    assert out["prompt_cache_retention"] == "in_memory"


def test_key_fields_are_the_options_not_the_key():
    assert OpenAIPromptCache().key_fields() == {"key": None, "retention": None}
    fields = OpenAIPromptCache(key=test_key_fields_are_the_options_not_the_key).key_fields()
    assert fields["key"].startswith("callable:tests.cache.providers.test_openai.")
    assert OpenAIPromptCacheV2(mode="explicit", ttl="12h", key="k").key_fields() == {
        "mode": "explicit",
        "ttl": "12h",
        "key": "k",
    }


@pytest.mark.parametrize(
    "make",
    [
        lambda: OpenAIPromptCache(retention="1d"),
        lambda: OpenAIPromptCacheV2(mode="auto"),
        lambda: OpenAIPromptCacheV2(ttl="1h"),
    ],
)
def test_bad_options_are_rejected(make):
    with pytest.raises(ConfigError):
        make()


async def test_v2_gate():
    assert await apply(OpenAIPromptCacheV2(caps=lookup(V1_24H))) == KWARGS
    out = await apply(OpenAIPromptCacheV2(ttl="12h", caps=lookup(V2)), model="gpt-5.6")
    assert out == {
        **KWARGS,
        "prompt_cache_options": {"mode": "implicit", "ttl": "12h"},
        "prompt_cache_key": digest("You are terse."),
    }


async def test_v2_implicit_mode_places_no_breakpoints():
    out = await apply(OpenAIPromptCacheV2(caps=lookup(V2)))
    assert out["messages"] == KWARGS["messages"]


BP = {"mode": "explicit"}


async def test_v2_explicit_marks_breakpoints_on_chat_messages():
    request = LLMRequest(
        messages=(
            SYSTEM,
            Message(role="user", content="same"),
            Message(role="assistant", content="ok"),
            Message(role="user", content="same", cache_breakpoint=True),
            Message(role="tool", content="r", tool_call_id="c1", cache_breakpoint=True),
        )
    )
    kwargs = {
        "messages": [
            {"role": "developer", "content": "You are terse."},
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "same"},
            {"role": "tool", "tool_call_id": "c0", "content": "r"},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ]
    }
    out = await apply(OpenAIPromptCacheV2(mode="explicit", caps=lookup(V2)), kwargs, request)
    assert out["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert out["messages"] == [
        {
            "role": "developer",
            "content": [{"type": "text", "text": "You are terse.", "prompt_cache_breakpoint": BP}],
        },
        {"role": "user", "content": "same"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": "same", "prompt_cache_breakpoint": BP}],
        },
        {"role": "tool", "tool_call_id": "c0", "content": "r"},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [{"type": "text", "text": "r", "prompt_cache_breakpoint": BP}],
        },
    ]


async def test_v2_explicit_marks_only_input_blocks_in_responses(caplog):
    request = LLMRequest(
        messages=(
            SYSTEM,
            Message(role="assistant", content="earlier", cache_breakpoint=True),
            Message(role="tool", content="42", tool_call_id="c", cache_breakpoint=True),
            Message(role="user", content="now", cache_breakpoint=True),
        )
    )
    kwargs = {
        "instructions": "You are terse.",
        "input": [
            {"role": "assistant", "content": "earlier"},
            {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": "42"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "now"}]},
        ],
    }
    plugin = OpenAIPromptCacheV2(mode="explicit", caps=lookup(V2))
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.cache.providers.openai"):
        out = await apply(plugin, kwargs, request)
    assert out["instructions"] == "You are terse."
    assert out["input"][0] == kwargs["input"][0]  # output_text cannot carry a breakpoint
    assert out["input"][1] == kwargs["input"][1]
    assert out["input"][2] == {
        "type": "function_call_output",
        "call_id": "c",
        "output": [{"type": "input_text", "text": "42", "prompt_cache_breakpoint": BP}],
    }
    assert out["input"][3]["content"] == [
        {"type": "input_text", "text": "now", "prompt_cache_breakpoint": BP}
    ]
    assert "assistant output" in caplog.text
    assert "2 cache breakpoint(s) had no content block" in caplog.text


async def test_v2_explicit_without_breakpoints_falls_back_to_implicit_and_warns_once(caplog):
    plugin = OpenAIPromptCacheV2(mode="explicit", ttl="12h", caps=lookup(V2))
    request = LLMRequest(messages=(Message(role="user", content="hi"),))
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cache.providers.openai"):
        for kwargs in ({"input": "plain text"}, {"messages": [{"role": "user", "content": "hi"}]}):
            out = await apply(plugin, kwargs, request)
            assert out["prompt_cache_options"] == {"ttl": "12h"}  # mode omitted: implicit
    assert caplog.text.count("no cache breakpoint") == 1
    assert out["messages"] == [{"role": "user", "content": "hi"}]


async def test_v2_keeps_host_prompt_cache_options():
    kwargs = {**KWARGS, "prompt_cache_options": {"mode": "implicit"}}
    out = await apply(OpenAIPromptCacheV2(mode="explicit", caps=lookup(V2)), kwargs)
    assert out["prompt_cache_options"] == {"mode": "implicit"}


async def test_v2_sets_no_key_without_a_system_message():
    request = LLMRequest(messages=(Message(role="user", content="hi"),))
    out = await apply(OpenAIPromptCacheV2(caps=lookup(V2)), request=request)
    assert "prompt_cache_key" not in out and "prompt_cache_options" in out


async def test_v2_explicit_places_at_most_four_breakpoints_latest_win(caplog):
    tools = [
        Message(role="tool", content=f"r{i}", tool_call_id=f"c{i}", cache_breakpoint=True)
        for i in range(5)
    ]
    request = LLMRequest(messages=(SYSTEM, *tools))
    kwargs = {
        "messages": [
            {"role": "system", "content": "You are terse."},
            *({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"} for i in range(5)),
        ]
    }
    plugin = OpenAIPromptCacheV2(mode="explicit", caps=lookup(V2))
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.cache.providers.openai"):
        out = await apply(plugin, kwargs, request)
    marked = [isinstance(e["content"], list) for e in out["messages"]]
    assert marked == [False, False, True, True, True, True]  # system and c0 dropped
    assert "2 breakpoint(s) dropped" in caplog.text


async def test_v2_explicit_counts_host_breakpoints_towards_the_cap():
    host = {"type": "text", "text": "h", "prompt_cache_breakpoint": BP}
    request = LLMRequest(
        messages=tuple(
            Message(role="user", content=f"u{i}", cache_breakpoint=True) for i in range(3)
        )
    )
    kwargs = {
        "messages": [
            {"role": "user", "content": [host]},
            {"role": "user", "content": [host]},
            *({"role": "user", "content": f"u{i}"} for i in range(3)),
        ]
    }
    out = await apply(OpenAIPromptCacheV2(mode="explicit", caps=lookup(V2)), kwargs, request)
    marked = [isinstance(e["content"], list) for e in out["messages"][2:]]
    assert marked == [False, True, True]
    assert out["prompt_cache_options"]["mode"] == "explicit"
