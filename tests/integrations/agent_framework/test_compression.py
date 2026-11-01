"""The agent_framework compression format on real MAF `Message`s, and the middleware example."""

import asyncio
from typing import Any

import pytest

pytest.importorskip("agent_framework")

from agent_framework import (
    Agent,
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
)

from agent_adapter.context.formats import convert
from agent_adapter.core.errors import ConfigError
from agent_adapter.integrations.agent_framework import AgentFrameworkConverter
from tests.integrations.compression_oracle import (
    DENIED,
    PAD,
    STEPS,
    check_round_trip,
    check_safe,
    example,
)

pytestmark = pytest.mark.contract


def call(call_id: str) -> Content:
    return Content.from_function_call(call_id, "lookup", arguments={"q": call_id})


def result(call_id: str, value: Any) -> Content:
    return Content.from_function_result(call_id, result=value)


def conversation() -> list[Message]:
    approval = Content.from_function_approval_request("a1", call("c5"))
    return [
        Message("system", ["Be brief."]),
        Message("user", [f"old question {PAD}"], author_name="ann"),
        Message("assistant", [Content.from_text_reasoning(text="hm"), f"look {PAD}", call("c1")]),
        Message("tool", [result("c1", f"found {PAD}")]),
        Message("assistant", [call("c2"), call("c3")], message_id="m4"),
        Message("tool", [result("c2", DENIED), result("c3", f"two {PAD}")]),
        Message("assistant", [approval]),
        Message("user", [approval.to_function_approval_response(approved=False)]),
        Message("assistant", [f"old answer {PAD}"]),
        Message("user", [f"middle {PAD}"]),
        Message("assistant", [call("c4")]),
        Message("tool", [result("c4", {"rows": PAD})]),
        Message("assistant", ["answer"]),
        Message("user", ["new question"]),
    ]


def test_round_trip_gives_equal_values_and_the_same_objects():
    check_round_trip(conversation(), "agent_framework", dump=lambda m: m.to_dict())


def test_ids_results_and_approval_messages():
    messages = convert(conversation()).messages
    assert [c.id for m in messages for c in m.tool_calls] == ["c1", "c2", "c3", "c4"]
    assert [m.tool_call_id for m in messages if m.role == "tool"] == ["c1", "c2", "c3", "c4"]
    assert messages[2].content == f"look {PAD}"  # reasoning is not text
    assert [m.pinned for m in messages[7:9]] == [True, True]  # approval request and response


@pytest.mark.parametrize("step", list(STEPS))
def test_no_step_orphans_a_tool_round_or_touches_protected_messages(step):
    items = conversation()
    smaller = check_safe(items, STEPS[step](), pinned=lambda m: m.author_name == "ann")
    kept = {id(m) for m in smaller}
    assert {id(items[n]) for n in (1, 4, 5, 6, 7)} <= kept  # pinned, denied round, approvals


def test_cut_keeps_contents_and_fields():
    items = conversation()
    smaller = check_safe(items, STEPS["cut"]())
    assert smaller[1].author_name == "ann" and "characters cut" in smaller[1].text
    assert [c.type for c in smaller[2].contents] == ["text_reasoning", "text", "function_call"]
    (cut,) = smaller[3].contents
    assert cut.call_id == "c1" and "characters cut" in cut.result
    assert [c.text for c in cut.items] == [cut.result]
    assert items[3].contents[0].result == f"found {PAD}"  # the input is never changed


def test_new_messages_render_as_maf_messages():
    smaller = check_safe(conversation(), STEPS["summary"]())
    assert any(m.role == "user" and m.text.startswith("summary") for m in smaller)
    rendered = AgentFrameworkConverter([]).render(convert(conversation()).messages)
    assert [c.type for c in rendered[2].contents] == ["text", "function_call"]
    assert rendered[2].contents[1].call_id == "c1"
    assert rendered[3].role == "tool" and rendered[3].contents[0].call_id == "c1"


def test_a_dict_is_not_an_agent_framework_message():
    with pytest.raises(ConfigError, match="not a agent_framework message"):
        convert([{"role": "user", "content": "a"}], format="agent_framework")


class RecordingClient(ChatMiddlewareLayer, FunctionInvocationLayer, BaseChatClient):  # type: ignore[misc]
    """A chat client with MAF's middleware layer, as its provider clients have; keeps each call."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[list[Message]] = []

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.seen.append(list(messages))
        return ChatResponse(messages=[Message("assistant", ["ok"])])


def test_docs_middleware_shortens_each_model_call():
    client = RecordingClient()
    compress_messages = example("compression_agent_framework")["compress_messages"]
    agent = Agent(client=client, middleware=[compress_messages])
    history = [m for _ in range(8) for m in conversation()[8:10]]
    response = asyncio.run(agent.run([*history, Message("user", ["new question"])]))
    assert response.text == "ok"
    (sent,) = client.seen
    assert len(sent) < len(history) and sent[-1].text == "new question"
