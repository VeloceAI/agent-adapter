"""The "pydantic_ai" format and its history processor, on real Pydantic AI messages."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ImageUrl,
    InstructionDeltaPart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_adapter.context.conversation import Protection
from agent_adapter.context.formats import compress_for_sync, convert
from agent_adapter.core.errors import ConfigError, MissingDependency
from agent_adapter.core.messages import Message
from agent_adapter.integrations.pydantic_ai import pydantic_ai_history_processor
from agent_adapter.integrations.pydantic_ai_compress import (
    PydanticAIConverter,
    _messages_module,
    _tool_text,
)
from tests.integrations.compression_support import (
    CUT,
    DROP_OLD,
    EXPECTED,
    PAD,
    assert_cut_keeps_types,
    assert_drop_keeps_whole_rounds,
    assert_round_trip,
    assert_summary_is_native,
    render_read_back,
)

pytestmark = pytest.mark.contract
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "compression_framework_hooks.py"


def history() -> list:
    return [
        ModelRequest(
            parts=[
                SystemPromptPart("Be brief."),
                UserPromptPart([f"old {PAD}", ImageUrl("https://x/a.png")]),
            ],
            instructions="Answer in English.",
        ),
        ModelResponse(
            parts=[
                ThinkingPart("hmm"),
                TextPart(f"let me look {PAD}"),
                ToolCallPart("lookup", {"q": "a"}, "c1"),
                ToolCallPart("lookup", '{"q": "b"}', "c2"),
            ],
            model_name="m",
        ),
        ModelRequest(
            parts=[
                ToolReturnPart("lookup", f"one {PAD}", "c1"),
                ToolReturnPart("lookup", {"v": 2}, "c2"),
            ]
        ),
        ModelResponse(parts=[TextPart(f"answer {PAD}")]),
        ModelRequest(parts=[UserPromptPart("new question")]),
    ]


def test_round_trip_keeps_parts_ids_and_roles() -> None:
    messages = assert_round_trip(history(), "pydantic_ai")
    roles = [m.role for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant", "user"]
    assert [c.id for c in messages[2].tool_calls] == ["c1", "c2"]
    assert messages[2].tool_calls[1].arguments == {"q": "b"}
    assert [(m.tool_call_id, m.name) for m in messages[3:5]] == [("c1", "lookup"), ("c2", "lookup")]
    assert "hmm" not in messages[2].content  # thinking is not text a step changes


def test_dropping_old_turns_never_orphans_a_tool_round() -> None:
    items = history()
    smaller = assert_drop_keeps_whole_rounds(items, whole=False)
    assert smaller[0].parts == [items[0].parts[0]]  # the system part stays, the old question goes
    assert smaller[0].instructions == "Answer in English." and smaller[1:] == [items[-1]]


def test_cut_keeps_media_thinking_calls_and_fields() -> None:
    items = history()
    smaller = assert_cut_keeps_types(items)
    first = smaller[0]
    assert first.instructions == "Answer in English." and first.parts[0] is items[0].parts[0]
    text, image = first.parts[1].content
    assert "characters cut" in text and image == ImageUrl("https://x/a.png")
    response = smaller[1]
    assert response.parts[0] is items[1].parts[0] and response.parts[2:] == items[1].parts[2:]
    assert response.model_name == "m" and "characters cut" in response.parts[1].content
    assert smaller[2].parts[0].tool_call_id == "c1" and smaller[2].parts[1] is items[2].parts[1]


def test_summary_becomes_a_model_request() -> None:
    smaller = assert_summary_is_native(history(), (ModelRequest, ModelResponse))
    assert any(
        isinstance(m, ModelRequest) and m.parts[0].content == "summary of earlier turns"
        for m in smaller
    )


def test_a_denied_tool_return_is_pinned() -> None:
    items = history()
    items[2] = ModelRequest(
        parts=[ToolReturnPart("lookup", "no", "c1", outcome="denied"), *items[2].parts[1:]]
    )
    messages = convert(items).messages
    assert messages[3].pinned and Protection(messages).is_protected(messages[3])
    smaller, _ = compress_for_sync(items, CUT)
    assert smaller[2].parts[0] is items[2].parts[0]


def test_a_legacy_retry_naming_its_tool_is_a_tool_result() -> None:
    with pytest.warns(Warning, match="deprecated"):
        retry = RetryPromptPart("bad args", tool_name="lookup", tool_call_id="c1")
    request = ModelRequest(parts=[retry])
    messages = convert([*history()[:2], request]).messages
    assert messages[-1].role == "tool" and messages[-1].tool_call_id == "c1"


async def test_history_processor_shortens_what_the_model_gets() -> None:
    seen: list[int] = []

    def respond(messages: list, info: AgentInfo) -> ModelResponse:
        seen.append(len(messages))
        return ModelResponse(parts=[TextPart("ok")])

    processor = pydantic_ai_history_processor(DROP_OLD)
    agent = Agent(FunctionModel(respond), capabilities=[ProcessHistory(processor)])
    result = await agent.run("newest", message_history=history())
    assert result.output == "ok" and seen == [1]  # 6 messages before compression


async def test_the_example_builds_the_agent() -> None:
    example = runpy.run_path(str(EXAMPLE))
    agent = example["pydantic_ai_agent"](
        FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("ok")]))
    )
    assert (await agent.run("hi")).output == "ok"


def test_new_messages_render_as_requests_and_responses() -> None:
    assert render_read_back(PydanticAIConverter) == EXPECTED


def test_cut_puts_the_text_in_the_first_user_part_and_keeps_later_media() -> None:
    image = ImageUrl("https://x/b.png")
    request = ModelRequest(parts=[UserPromptPart(f"a {PAD}"), UserPromptPart([f"b {PAD}", image])])
    items = [*history()[:4], request, history()[-1]]
    smaller, _ = compress_for_sync(items, CUT)
    first, second = smaller[4].parts
    assert "characters cut" in first.content and second.content == [image]


def test_an_item_that_is_not_a_pydantic_ai_message_is_refused() -> None:
    with pytest.raises(ConfigError, match="item 1 is not a pydantic_ai message: dict"):
        convert([history()[0], {"role": "user", "content": "x"}], format="pydantic_ai")


def test_a_legacy_retry_without_a_tool_is_user_feedback() -> None:
    with pytest.warns(Warning, match="deprecated"):
        retry = RetryPromptPart("try again")
    messages = convert([*history()[:2], ModelRequest(parts=[retry])]).messages
    assert messages[-1].role == "user" and "try again" in messages[-1].content


def test_a_changed_request_keeps_its_state_parts_and_later_media() -> None:
    delta = InstructionDeltaPart(id="i1", content="Be terse.")
    image = ImageUrl("https://x/c.png")
    request = ModelRequest(parts=[UserPromptPart("a"), delta, UserPromptPart(["b", image])])
    parts = PydanticAIConverter([request]).retext_parts(
        request, list(request.parts), Message(role="user", content="new")
    )
    assert parts == [UserPromptPart("new"), delta, UserPromptPart([image])]


def test_new_response_text_goes_first_when_the_response_had_none() -> None:
    response = ModelResponse(parts=[ToolCallPart("lookup", {"q": "a"}, "c1")])
    changed = PydanticAIConverter([response]).retext(
        response, Message(role="assistant", content="hi")
    )
    assert changed.parts == [TextPart("hi"), *response.parts]


def test_a_part_without_a_renderer_reads_its_content_as_json() -> None:
    assert _tool_text(SimpleNamespace(content={"b": 1, "a": 2})) == '{"a": 2, "b": 1}'


def test_building_messages_without_pydantic_ai_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[pydantic-ai\]"):
        _messages_module()
