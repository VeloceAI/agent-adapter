"""Memory events carry the scope kind (which fields are set), never the scope ids."""

from __future__ import annotations

from pathlib import Path

from agent_adapter.core.context import ToolContext
from agent_adapter.memory import MemoryScope, SQLiteMemory
from agent_adapter.memory.records import scope_kind
from agent_adapter.tracing.sinks import EventLog
from tests.memory.helpers import ALICE, ALICE_ALL, BOB, rec


def test_scope_kind_names_fields_never_ids() -> None:
    assert scope_kind([ALICE]) == "user+session"
    assert scope_kind([ALICE_ALL]) == "user"
    assert scope_kind([MemoryScope(agent_id="a1")]) == "agent"
    assert scope_kind([ALICE, BOB, ALICE_ALL]) == "user,user+session"
    assert scope_kind([None]) == "ids" and scope_kind([]) == "ids"


async def test_every_memory_event_has_a_scope_kind(tmp_path: Path) -> None:
    log = EventLog()
    ctx = ToolContext(run_id="run_1", sink=log)
    memory = SQLiteMemory(tmp_path / "m.db")
    (stored,) = await memory.add([rec(ALICE, "likes trains")], ctx=ctx)
    await memory.search("trains", scope=ALICE_ALL, ctx=ctx)
    await memory.export(ALICE_ALL, ctx=ctx)
    await memory.delete([stored.id], ctx=ctx)
    await memory.delete(scope=ALICE_ALL, ctx=ctx)
    kinds = [(event.type, event.data["scope_kind"]) for event in log.of_type("memory.")]
    assert kinds == [
        ("memory.added", "user+session"),
        ("memory.searched", "user"),
        ("memory.exported", "user"),
        ("memory.deleted", "ids"),
        ("memory.deleted", "user"),
    ]
    assert "alice" not in repr(log.events)
