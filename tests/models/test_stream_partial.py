"""Streamed structured output: each port follows JSON text deltas with `PartialOutput` events."""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent_adapter.core.messages import (
    LLMRequest,
    LLMResponse,
    Message,
    OutputSchema,
    PartialOutput,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
)
from agent_adapter.models import CallableModel
from agent_adapter.models._port import replay
from tests.models.conftest import fixture

SCHEMA = OutputSchema(
    name="Ticket",
    json_schema={
        "type": "object",
        "properties": {"title": {"type": "string"}, "seats": {"type": "integer"}},
    },
)
PIECES = ['{"title": "Mu', 'seum tour", "se', 'ats": 4', "}"]
WHOLE = {"title": "Museum tour", "seats": 4}


def _request(**fields: Any) -> LLMRequest:
    return LLMRequest(messages=(Message(role="user", content="book"),), **fields)


def _partials(events: list[Any]) -> list[Any]:
    return [e.value for e in events if isinstance(e, PartialOutput)]


def _check(events: list[Any]) -> None:
    assert [e.text for e in events if isinstance(e, TextDelta)] == PIECES
    assert _partials(events) == [
        {"title": "Mu"},
        {"title": "Museum tour"},  # "se is an unfinished key: left out
        WHOLE,
    ]
    for index, event in enumerate(events):
        if isinstance(event, PartialOutput):
            assert isinstance(events[index - 1], TextDelta)
    done = events[-1]
    assert isinstance(done, StreamDone) and done.response.parsed == WHOLE


def _openai_chat() -> list[dict[str, Any]]:
    base = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4.1"}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]}
        for p in PIECES
    ]
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    return chunks


def _openai_responses() -> list[dict[str, Any]]:
    recorded = fixture("openai/responses_stream")
    created, completed = recorded[0], copy.deepcopy(recorded[-1])
    deltas = [
        {
            "type": "response.output_text.delta",
            "sequence_number": n + 1,
            "output_index": 0,
            "item_id": "msg_1",
            "content_index": 0,
            "delta": p,
            "logprobs": [],
        }
        for n, p in enumerate(PIECES)
    ]
    message = completed["response"]["output"][0]
    message["content"][0]["text"] = "".join(PIECES)
    completed["response"]["output"] = [message]
    return [created, *deltas, completed]


async def test_openai_chat_streams_partial_output() -> None:
    port = pytest.importorskip("tests.models.test_openai_port")
    wire = port.Wire(port.sse_chat(_openai_chat()))
    model = port.OpenAIModel("gpt-4.1", api="chat", client=port.client_for(wire))
    _check([e async for e in model.stream(_request(output_schema=SCHEMA))])


async def test_openai_responses_streams_partial_output() -> None:
    port = pytest.importorskip("tests.models.test_openai_port")
    wire = port.Wire(port.sse_responses(_openai_responses()))
    model = port.OpenAIModel("gpt-5", client=port.client_for(wire))
    _check([e async for e in model.stream(_request(output_schema=SCHEMA))])


async def test_anthropic_streams_partial_output() -> None:
    port = pytest.importorskip("tests.models.test_anthropic_port")
    recorded = fixture("anthropic/stream")
    start, stop = recorded[0], recorded[-1]
    events = [
        start,
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        *[
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": p}}
            for p in PIECES
        ],
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
        stop,
    ]
    model = port.AnthropicModel("claude-test", client=port.client_for(port.Wire(port.sse(events))))
    _check([e async for e in model.stream(_request(output_schema=SCHEMA))])


async def test_gemini_streams_partial_output() -> None:
    port = pytest.importorskip("tests.models.test_gemini_port")
    chunks: list[dict[str, Any]] = [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": p}]}, "index": 0}]}
        for p in PIECES
    ]
    chunks[-1]["candidates"][0]["finishReason"] = "STOP"
    chunks[-1]["usageMetadata"] = {"promptTokenCount": 3, "candidatesTokenCount": 9}
    model = port.GeminiModel(
        "gemini-2.5-flash", client=port.client_for(port.Wire(port.sse(chunks)))
    )
    _check([e async for e in model.stream(_request(output_schema=SCHEMA))])


async def test_without_a_schema_no_partial_output_and_tool_deltas_unchanged() -> None:
    port = pytest.importorskip("tests.models.test_openai_port")
    wire = port.Wire(port.sse_chat(fixture("openai/chat_stream")))
    model = port.OpenAIModel("gpt-4.1", api="chat", client=port.client_for(wire))
    events = [e async for e in model.stream(_request())]
    assert _partials(events) == []
    deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert deltas[0] == ToolCallDelta(index=0, id="call_a", name="get_weather")
    assert "".join(d.arguments_delta for d in deltas if d.index == 0) == '{"city":"Paris"}'


async def test_text_that_is_not_json_has_no_partial_output() -> None:
    port = pytest.importorskip("tests.models.test_openai_port")
    base = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4.1"}
    chunks = [
        {
            **base,
            "choices": [{"index": 0, "delta": {"content": "Sorry, no."}, "finish_reason": "stop"}],
        }
    ]
    model = port.OpenAIModel(
        "gpt-4.1", api="chat", client=port.client_for(port.Wire(port.sse_chat(chunks)))
    )
    with pytest.raises(Exception) as caught:  # the final parse fails as before
        _ = [e async for e in model.stream(_request(output_schema=SCHEMA))]
    assert type(caught.value).__name__ == "ModelOutputInvalid"


def test_replay_carries_the_parsed_output() -> None:
    response = LLMResponse(
        provider="callable",
        model="m",
        text=json.dumps(WHOLE),
        parsed=WHOLE,
        message=Message(role="assistant", content=json.dumps(WHOLE)),
    )
    kinds = [type(e).__name__ for e in replay(response)]
    assert kinds == ["TextDelta", "PartialOutput", "StreamDone"]
    assert _partials(replay(response)) == [WHOLE]


async def test_callable_model_replays_partial_output_when_it_cannot_stream() -> None:
    def answer(request: LLMRequest) -> str:
        return json.dumps(WHOLE)

    model = CallableModel(answer)
    events = [e async for e in model.stream(_request(output_schema=SCHEMA))]
    assert _partials(events) == [WHOLE]


async def test_callable_stream_fn_text_gets_partial_output() -> None:
    async def events(request: LLMRequest) -> AsyncIterator[StreamEvent]:
        for piece in PIECES:
            yield TextDelta(text=piece)
        yield StreamDone(response=_parsed_response())

    model = CallableModel(lambda request: "unused", stream_fn=events)
    _check([e async for e in model.stream(_request(output_schema=SCHEMA))])
    plain = [e async for e in model.stream(_request())]
    assert _partials(plain) == []


async def test_a_stream_fn_with_its_own_partial_output_is_left_alone() -> None:
    async def events(request: LLMRequest) -> AsyncIterator[StreamEvent]:
        yield PartialOutput(value={"title": "own"})
        for piece in PIECES:
            yield TextDelta(text=piece)
        yield StreamDone(response=_parsed_response())

    model = CallableModel(lambda request: "unused", stream_fn=events)
    events_seen = [e async for e in model.stream(_request(output_schema=SCHEMA))]
    assert _partials(events_seen) == [{"title": "own"}]


def _parsed_response() -> LLMResponse:
    body = "".join(PIECES)
    return LLMResponse(
        provider="callable",
        model="host",
        text=body,
        parsed=WHOLE,
        message=Message(role="assistant", content=body),
    )
