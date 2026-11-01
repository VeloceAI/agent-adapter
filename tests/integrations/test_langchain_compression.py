"""The "langchain" format on real langchain_core messages, with no LangGraph installed."""

import runpy
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent_adapter.context.formats import compress_for_sync, convert
from agent_adapter.context.formats.langchain import LangChainConverter
from tests.integrations.compression_support import (
    DENIED,
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
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "compression_langchain.py"


def history() -> list[BaseMessage]:
    calls = [
        {"name": "lookup", "args": {"q": "a"}, "id": "c1", "type": "tool_call"},
        {"name": "lookup", "args": {"q": "b"}, "id": "c2", "type": "tool_call"},
    ]
    return [
        SystemMessage(content="Be brief.", id="s1"),
        HumanMessage(content=f"old {PAD}", id="h1", additional_kwargs={"keep": True}),
        AIMessage(content=[{"type": "text", "text": f"look {PAD}"}], tool_calls=calls, id="a1"),
        ToolMessage(content=f"one {PAD}", tool_call_id="c1", name="lookup", id="t1"),
        ToolMessage(content=f"two {PAD}", tool_call_id="c2", name="lookup", id="t2"),
        AIMessage(content=f"answer {PAD}", id="a2", response_metadata={"model": "m"}),
        HumanMessage(content="new question", id="h2"),
    ]


def test_round_trip_keeps_ids_roles_and_calls() -> None:
    messages = assert_round_trip(history(), "langchain")
    roles = [m.role for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant", "user"]
    assert [c.id for c in messages[2].tool_calls] == ["c1", "c2"]
    assert [m.tool_call_id for m in messages[3:5]] == ["c1", "c2"]


def test_dropping_old_turns_never_orphans_a_tool_round() -> None:
    smaller = assert_drop_keeps_whole_rounds(history())
    assert smaller[-1].content == "new question"


def test_cut_keeps_ids_blocks_and_kwargs() -> None:
    items = history()
    smaller = assert_cut_keeps_types(items)
    human, ai, answer = smaller[1], smaller[2], smaller[5]
    assert human.id == "h1" and human.additional_kwargs == {"keep": True}
    assert ai.tool_calls == items[2].tool_calls and ai.content[0]["type"] == "text"
    assert answer.response_metadata == {"model": "m"} and "characters cut" in answer.content


def test_a_summary_becomes_a_real_langchain_message() -> None:
    assert_summary_is_native(history(), BaseMessage)


def test_a_denied_approval_result_is_kept_verbatim() -> None:
    items = history()
    items[3] = ToolMessage(content=DENIED, tool_call_id="c1", name="lookup", id="t1")
    smaller, _ = compress_for_sync(items, DROP_OLD)
    assert any(s is items[3] for s in smaller)


def test_pinned_messages_are_kept_and_not_marked() -> None:
    items = history()
    smaller, _ = compress_for_sync(
        items, DROP_OLD, pinned=lambda m: bool(m.additional_kwargs.get("keep"))
    )
    assert smaller[1] is items[1] and items[1].additional_kwargs == {"keep": True}


def test_new_messages_render_as_langchain_messages() -> None:
    assert render_read_back(LangChainConverter) == EXPECTED


def test_the_chat_model_gets_the_shorter_list() -> None:
    llm = FakeListChatModel(responses=["ok"])
    smaller, report = compress_for_sync(history(), DROP_OLD)
    assert report.tokens_after < report.tokens_before
    assert llm.invoke(smaller).content == "ok"
    assert convert(smaller).messages[-1].content == "new question"


def test_the_example_compresses_before_invoke(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "langchain: 8 messages, 4 sent",
        "reply: The rate is 30%.",
    ]
