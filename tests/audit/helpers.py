"""Shared builders for the audit tests."""

from __future__ import annotations

from pathlib import Path

from agent_adapter.audit import AuditEntry, FileAuditStore

KEY = "museum-audit-key-0123456789abcdef"


def entry(
    action: str = "approval.requested", ts: float = 1_800_000_000.0, **data: object
) -> AuditEntry:
    """An entry with a tool name unless data says otherwise."""
    return AuditEntry(action, ts, "run_1", "call_1", data or {"tool": "refund"})


def write_log(folder: Path, count: int = 3, **options: object) -> FileAuditStore:
    """A closed store in `folder` holding `count` entries."""
    store = FileAuditStore(folder, key=KEY, fsync=False, **options)  # type: ignore[arg-type]
    for n in range(count):
        store.write(entry(ts=1_800_000_000.0 + n))
    store.close()
    return store
