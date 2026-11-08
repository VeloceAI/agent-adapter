"""memory_tools through the Executor: scope from the context, guardrails, approvals, events."""

import pytest

from agent_adapter import Executor, Registry
from agent_adapter.core.context import ToolContext
from agent_adapter.errors import (
    ApprovalDenied,
    ConfigError,
    GuardrailBlocked,
    ToolArgsInvalid,
    ToolError,
)
from agent_adapter.hooks.builtin.guards import PIIGuard
from agent_adapter.memory import (
    MemoryScope,
    SQLiteMemory,
    memory_tools,
    scope_from_context,
    thread_scope,
    user_scope,
)
from agent_adapter.tracing.sinks import CONTENT_KEYS
from tests.fakes.sinks import RecordingSink
from tests.memory.helpers import embedder, rec


def _setup(tmp_path, *, hooks=(), sink=None, memory=None, **options):
    memory = memory or SQLiteMemory(tmp_path / "tools.db")
    registry = Registry()
    registry.add_many(memory_tools(memory, **options))
    return Executor(registry, hooks=hooks, sink=sink), memory


def _ctx(executor, user="alice", thread="chat-1", **kw):
    ctx = executor.context(thread_id=thread, **kw)
    return ToolContext(**{**ctx.__dict__, "user_id": user})


async def test_remember_then_recall_across_chats_of_the_same_user(tmp_path):
    executor, memory = _setup(tmp_path)
    saved = await executor.run("remember", {"text": "Prefers aisle seats"}, _ctx(executor))
    assert saved.startswith("Saved as mem_")
    found = await executor.run("recall", {"query": "seats"}, _ctx(executor, thread="chat-2"))
    assert [item["text"] for item in found] == ["Prefers aisle seats"]
    assert set(found[0]) == {"id", "text", "saved"}
    (record,) = await memory.export(MemoryScope(user_id="alice"))
    assert record.scope == MemoryScope(user_id="alice", session_id="chat-1")
    assert record.source.startswith("run_")


async def test_another_user_recalls_nothing_and_cannot_forget(tmp_path):
    executor, _ = _setup(tmp_path)
    await executor.run("remember", {"text": "Alice's locker is 42"}, _ctx(executor))
    bob = _ctx(executor, user="bob")
    assert await executor.run("recall", {"query": "locker"}, bob) == []
    (item,) = await executor.run("recall", {"query": "locker"}, _ctx(executor))
    with pytest.raises(ToolError):
        await executor.run("forget", {"id": item["id"]}, bob)
    assert (
        await executor.run("forget", {"id": item["id"]}, _ctx(executor)) == f"Deleted {item['id']}."
    )


async def test_the_model_cannot_pass_a_scope(tmp_path):
    executor, _ = _setup(tmp_path)
    for spec in memory_tools(SQLiteMemory(tmp_path / "x.db")):
        props = spec.args_model.model_json_schema()["properties"]
        assert not {"scope", "user_id", "session_id", "ctx"} & set(props), spec.name
    with pytest.raises(ToolArgsInvalid):
        await executor.run("recall", {"query": "x", "user_id": "bob"}, _ctx(executor))


async def test_a_context_without_user_or_thread_fails_closed(tmp_path):
    executor, _ = _setup(tmp_path)
    with pytest.raises(ToolError):
        await executor.run("remember", {"text": "x"}, _ctx(executor, user="", thread=""))
    with pytest.raises(ToolError):
        await executor.run("recall", {"query": ""}, _ctx(executor, user="", thread=""))


async def test_without_a_user_only_the_same_thread_is_seen(tmp_path):
    executor, _ = _setup(tmp_path, scope_from=thread_scope)
    await executor.run("remember", {"text": "guest wants a map"}, _ctx(executor, user=""))
    assert (
        await executor.run("recall", {"query": "map"}, _ctx(executor, user="", thread="other"))
        == []
    )
    assert len(await executor.run("recall", {"query": "map"}, _ctx(executor, user=""))) == 1


async def test_across_sessions_false_keeps_recall_in_the_thread(tmp_path):
    executor, _ = _setup(tmp_path, across_sessions=False)
    await executor.run("remember", {"text": "note"}, _ctx(executor))
    assert await executor.run("recall", {"query": "note"}, _ctx(executor, thread="t2")) == []


async def test_a_failing_scope_from_fails_closed_without_detail(tmp_path, caplog):
    def broken(ctx):
        raise KeyError("token=abc")

    executor, _ = _setup(tmp_path, scope_from=broken)
    with pytest.raises(ToolError):
        await executor.run("remember", {"text": "x"}, _ctx(executor))
    assert "token=abc" not in caplog.text


async def test_guardrails_see_remember_args_and_block_pii(tmp_path):
    executor, memory = _setup(tmp_path, hooks=[PIIGuard()])
    with pytest.raises(GuardrailBlocked):
        await executor.run("remember", {"text": "mail me at ann@example.com"}, _ctx(executor))
    assert await memory.export(MemoryScope(user_id="alice")) == []


async def test_guardrails_redact_recalled_results_before_the_model_sees_them(tmp_path):
    executor, memory = _setup(tmp_path, hooks=[PIIGuard(action="redact", on=("result",))])
    await memory.add([rec(MemoryScope(user_id="alice"), "mail ann@example.com")])
    (item,) = await executor.run("recall", {"query": "mail"}, _ctx(executor))
    assert "ann@example.com" not in item["text"]


async def test_confirm_writes_needs_an_approval(tmp_path):
    executor, _ = _setup(tmp_path, confirm_writes=True)
    with pytest.raises(ApprovalDenied):
        await executor.run("remember", {"text": "x"}, _ctx(executor))
    assert await executor.run("recall", {"query": ""}, _ctx(executor)) == []


async def test_forget_can_be_left_out_and_empty_text_is_refused(tmp_path):
    assert [s.name for s in memory_tools(SQLiteMemory(tmp_path / "a.db"), forget=False)] == [
        "remember",
        "recall",
    ]
    executor, _ = _setup(tmp_path)
    with pytest.raises(ToolError):
        await executor.run("remember", {"text": "   "}, _ctx(executor))


async def test_events_carry_counts_but_no_content(tmp_path):
    sink = RecordingSink()
    executor, _ = _setup(tmp_path, sink=sink)
    ctx = _ctx(executor)
    await executor.run("remember", {"text": "secret museum plan"}, ctx)
    await executor.run("recall", {"query": "museum"}, ctx)
    events = {e.type: e for e in sink.events if e.type.startswith("memory.")}
    assert set(events) == {"memory.added", "memory.searched"}
    assert events["memory.added"].data["count"] == 1
    assert events["memory.added"].run_id == ctx.run_id
    searched = events["memory.searched"].data
    assert (searched["backend"], searched["count"], searched["k"], searched["mode"]) == (
        "sqlite",
        1,
        5,
        "keyword",
    )
    assert "secret" not in repr([e.data for e in events.values()])


async def test_capture_content_puts_texts_under_content_keys(tmp_path):
    sink = RecordingSink()
    memory = SQLiteMemory(tmp_path / "c.db", capture_content=True, embedder=embedder())
    executor, _ = _setup(tmp_path, sink=sink, memory=memory)
    ctx = _ctx(executor)
    await executor.run("remember", {"text": "has a cat"}, ctx)
    await executor.run("recall", {"query": "kitten"}, ctx)
    (item,) = await executor.run("recall", {"query": ""}, ctx)
    await executor.run("forget", {"id": item["id"]}, ctx)
    by_type = [e for e in sink.events if e.type.startswith("memory.")]
    added, searched = by_type[0], by_type[1]
    assert added.data["arguments"] == {"texts": ["has a cat"]}
    assert searched.data["result"] == ["has a cat"] and searched.data["mode"] == "vector"
    assert {"arguments", "result"} <= CONTENT_KEYS
    assert by_type[-1].type == "memory.deleted" and "arguments" not in by_type[-1].data


def test_scope_helpers_read_user_principal_and_thread():
    assert scope_from_context(ToolContext()) is None
    assert scope_from_context(ToolContext(thread_id="t")) is None  # anonymous: no memory
    assert thread_scope(ToolContext(thread_id="t")) == MemoryScope(session_id="t")
    assert thread_scope(ToolContext()) is None
    assert thread_scope(ToolContext(user_id="u", thread_id="t")).user_id == "u"
    assert scope_from_context(ToolContext(thread_id="t", extras={"principal": ""})) is None
    assert scope_from_context(ToolContext(user_id="u", thread_id="t")) == MemoryScope(
        user_id="u", session_id="t"
    )

    class Principal:
        id = "p1"

    assert scope_from_context(ToolContext(extras={"principal": Principal()})).user_id == "p1"
    assert scope_from_context(ToolContext(extras={"principal": "p2"})).user_id == "p2"
    assert scope_from_context(ToolContext(extras={"principal": 7})) is None
    assert scope_from_context(ToolContext(user_id="u", extras={"principal": "p"})).user_id == "u"
    assert user_scope(ToolContext(user_id="u", thread_id="t")) == MemoryScope(user_id="u")
    assert user_scope(ToolContext(thread_id="t")) is None


@pytest.mark.parametrize("options", [{"k": 0}, {"k": True}, {"scope_from": "user"}])
def test_bad_tool_options_are_refused(tmp_path, options):
    with pytest.raises(ConfigError):
        memory_tools(SQLiteMemory(tmp_path / "x.db"), **options)
    with pytest.raises(ConfigError):
        memory_tools(object())


async def test_anonymous_callers_on_the_same_thread_get_no_memory_by_default(tmp_path):
    executor, _ = _setup(tmp_path)
    for _ in range(2):
        with pytest.raises(ToolError):
            await executor.run("recall", {"query": ""}, _ctx(executor, user="", thread="shared"))


async def test_user_only_backends_default_to_the_user_scope():
    from agent_adapter.memory import ZepMemory
    from tests.memory.fakes import FakeZep

    memory = ZepMemory(FakeZep())
    registry = Registry()
    registry.add_many(memory_tools(memory))
    executor = Executor(registry)
    await executor.run("remember", {"text": "likes jazz"}, _ctx(executor))
    found = await executor.run("recall", {"query": "jazz"}, _ctx(executor, thread="other"))
    assert [item["text"] for item in found] == ["likes jazz"]
