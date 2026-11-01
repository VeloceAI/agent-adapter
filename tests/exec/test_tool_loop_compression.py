"""run_tool_loop(compression=): the loop goes on with the compressed history, one overflow retry."""

import pytest

from agent_adapter import Registry
from agent_adapter.context import CompressionPolicy, RemoveOldMessages
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, ModelContextOverflow, ToolUsageError
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from agent_adapter.exec import Executor, run_tool_loop
from tests.fakes.models import ScriptedModel, calls, text
from tests.fakes.sinks import RecordingLedger, RecordingSink

registry = Registry()
PAD = "x" * 400


@registry.tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"{q}: {PAD}"


HISTORY = [
    Message(role="system", content="s"),
    Message(role="user", content=f"old {PAD}"),
    Message(role="assistant", content=f"old answer {PAD}"),
    Message(role="user", content="find a"),
]
STRONG = CompressionPolicy(
    start_when_tokens_over=10_000,
    shrink_to_tokens=5,
    steps=[RemoveOldMessages(keep_first=0, keep_last=1)],
)
POLICY = CompressionPolicy(
    start_when_tokens_over=150,
    shrink_to_tokens=100,
    steps=[RemoveOldMessages(keep_first=0, keep_last=1)],
    if_still_too_long=STRONG,
)


def _call(n: int) -> ToolCall:
    return ToolCall(name="lookup", arguments={"q": f"q{n}"}, id=f"c{n}")


async def test_history_is_compressed_before_each_call_and_kept_short():
    model = ScriptedModel([calls(_call(1)), calls(_call(2)), text("done")])
    sink = RecordingSink()
    result = await run_tool_loop(
        model, registry, HISTORY, compression=POLICY, executor=Executor(registry, sink=sink)
    )
    assert result.finish == "stop" and result.final.text == "done"
    first, second, third = (r.messages for r in model.requests)
    assert [m.role for m in first] == ["system", "user"] and first[-1].content == "find a"
    assert len(second) <= 4 and len(third) <= 4  # older rounds went as the history grew
    calls_sent = {c.id for m in third for c in m.tool_calls}
    results_sent = {m.tool_call_id for m in third if m.role == "tool"}
    assert calls_sent == results_sent  # no orphan ever reaches the model
    assert result.messages[: len(third)] == tuple(third)
    assert all(r.compression is None for r in model.requests)
    assert sink.of_type("context.compressed")


async def test_an_overflow_retries_once_with_the_stronger_policy():
    model = ScriptedModel(
        [ModelContextOverflow("too long", provider="scripted", status_code=400), text("ok")]
    )
    plenty = CompressionPolicy(
        start_when_tokens_over=10_000, shrink_to_tokens=9_000, if_still_too_long=STRONG
    )
    result = await run_tool_loop(model, registry, HISTORY, compression=plenty)
    assert result.final.text == "ok"
    assert len(model.requests[0].messages) == 4 and len(model.requests[1].messages) == 2


async def test_a_second_overflow_propagates():
    error = ModelContextOverflow("too long", provider="scripted", status_code=400)
    model = ScriptedModel([error, error])
    with pytest.raises(ModelContextOverflow):
        await run_tool_loop(model, registry, HISTORY, compression=POLICY)
    plain = CompressionPolicy(start_when_tokens_over=10_000, shrink_to_tokens=9_000)
    with pytest.raises(ModelContextOverflow):
        await run_tool_loop(ScriptedModel([error]), registry, HISTORY, compression=plain)


async def test_bad_settings_fail_before_anything_runs():
    model = ScriptedModel([])
    with pytest.raises(ConfigError, match="CompressionPolicy"):
        await run_tool_loop(model, registry, HISTORY, compression="small")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="no context window"):
        await run_tool_loop(model, registry, HISTORY, compression=CompressionPolicy())
    with pytest.raises(ToolUsageError, match="compression"):
        await run_tool_loop(model, registry, HISTORY, request_options={"compression": POLICY})
    assert model.requests == []


async def test_resuming_a_compressed_run_never_reuses_a_paid_entry_id():
    ledger = RecordingLedger()
    executor = Executor(registry, ledger=ledger)
    model = ScriptedModel([calls(_call(1)), calls(_call(2)), text("one")])
    first = await run_tool_loop(
        model, registry, HISTORY, compression=POLICY, executor=executor, ctx=ToolContext(run_id="r")
    )
    paid = {e.entry_id for e in ledger.entries("r") if e.kind == "llm"}
    assert len(paid) == 3
    later = [*first.messages, Message(role="user", content="and b?")]
    assert sum(1 for m in later if m.role == "assistant") < 3  # compression removed turns
    again = ScriptedModel([text("two")])
    await run_tool_loop(
        again, registry, later, compression=POLICY, executor=executor, ctx=ToolContext(run_id="r")
    )
    entries = [e.entry_id for e in ledger.entries("r") if e.kind == "llm"]
    assert len(entries) == 4 and len(set(entries)) == 4
