import json

import pytest

from agent_adapter import Message, Registry, ToolCall, run_tool_loop
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import LLMRequest, StreamDone, TextDelta
from agent_adapter.exec import Executor, HandoffModel, LoopAgent, handoff
from agent_adapter.exec.handoffs import HANDOFF_TAG, handed_off
from tests.exec.agent_helpers import Scripted, reply, shop


def front_desk(target_answer="billing says hi"):
    registry = Registry()
    executor = Executor(registry)
    registry.add(
        handoff(
            lambda task, ctx: target_answer,
            name="to_billing",
            description="Pass the customer to billing.",
        )
    )
    return registry, executor


async def test_a_handoff_ends_the_loop_with_the_target_answer():
    registry, executor = front_desk()
    model = Scripted("to_billing", {"task": "invoice question"})
    result = await run_tool_loop(
        HandoffModel(model, registry),
        registry,
        [Message(role="user", content="my invoice?")],
        executor=executor,
    )
    assert (result.finish, result.final.text) == ("stop", "billing says hi")
    assert len(model.requests) == 1  # no model call after the handoff
    assert result.final.usage.input_tokens == 0


async def test_a_loop_agent_target_shares_the_executor():
    registry = Registry()
    executor = Executor(registry)
    billing = LoopAgent(Scripted("stock", {"item": "lamp"}), shop().subset(names=["stock"]))
    registry.add(handoff(billing, name="to_stock", description="Stock desk.", executor=executor))
    result = await run_tool_loop(
        HandoffModel(Scripted("to_stock", {"task": "lamps"}), ["to_stock"]),
        registry,
        [Message(role="user", content="lamps?")],
        executor=executor,
    )
    assert result.final.text == "done: 3"
    assert HANDOFF_TAG in registry.get("to_stock").tags


async def test_streaming_handoff_yields_the_answer_once():
    registry, _ = front_desk()
    model = HandoffModel(Scripted(), registry)
    history = (
        Message(role="user", content="q"),
        Message(role="assistant", tool_calls=(ToolCall(name="to_billing", arguments={}, id="1"),)),
        Message(role="tool", tool_call_id="1", name="to_billing", content="from billing"),
    )
    events = [event async for event in model.stream(LLMRequest(messages=history))]
    assert isinstance(events[0], TextDelta) and events[0].text == "from billing"
    assert isinstance(events[1], StreamDone) and events[1].response.text == "from billing"
    assert (await model.generate(LLMRequest(messages=history))).text == "from billing"


async def test_without_a_handoff_the_wrapped_model_answers():
    registry, _ = front_desk()
    inner = Scripted()
    model = HandoffModel(inner, registry)
    request = LLMRequest(messages=(Message(role="user", content="hi"),))
    assert (await model.generate(request)).text == "answer to hi"
    streamed = [event async for event in model.stream(request)]
    assert streamed[-1].response.text == "answer to hi"
    assert model.cache_key_fields() == inner.cache_key_fields()
    assert (model.provider, model.model) == ("host", "scripted")


def test_handed_off_ignores_failures_and_older_turns():
    call = (ToolCall(name="h", arguments={}, id="1"),)
    error = json.dumps({"error": {"kind": "tool-error", "message": "x"}})
    failed = [
        Message(role="assistant", tool_calls=call),
        Message(role="tool", tool_call_id="1", name="h", content=error),
    ]
    assert handed_off(failed, ["h"]) is None
    older = [*failed[:1], Message(role="tool", name="h", content="ok"), reply("next").message]
    assert handed_off(older, ["h"]) is None
    assert handed_off([Message(role="user", content="q")], ["h"]) is None
    json_answer = [*failed[:1], Message(role="tool", name="h", content='{"error": 1, "x": 2}')]
    assert handed_off(json_answer, ["h"]) == '{"error": 1, "x": 2}'
    not_json = [*failed[:1], Message(role="tool", name="h", content='{"error" oops')]
    assert handed_off(not_json, ["h"]) == '{"error" oops'


def test_a_handoff_model_needs_handoffs():
    with pytest.raises(ConfigError, match="at least one"):
        HandoffModel(Scripted(), Registry())
