"""Which events are security-relevant, and the audit entries each one becomes."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.audit._scrub import Digest, clean_data, clean_value
from agent_adapter.core.types import Event
from agent_adapter.tracing.events import EventType as T

_BUDGET = ("scope", "window", "limit_usd", "spent_usd", "reserved_usd")
_MEMORY = ("backend", "count", "scope_kind", "scope")
_SANDBOX = ("sandbox", "backend", "language")
# Catalogued event -> (audit action, the data keys it keeps).
DIRECT: Mapping[str, tuple[str, tuple[str, ...]]] = {
    T.APPROVAL_REQUESTED: ("approval.requested", ("tool", "approval_id")),
    T.APPROVAL_EXPIRED: ("approval.expired", ("tool", "approval_id", "outcome")),
    T.GUARDRAIL_REDACTED: (
        "guardrail.redacted",
        ("guard", "stage", "count", "kinds", "tool", "provider", "model"),
    ),
    T.BUDGET_RESERVED: ("budget.reserved", _BUDGET),
    T.BUDGET_WARNING: ("budget.warning", _BUDGET),
    T.BUDGET_EXCEEDED: ("budget.exceeded", _BUDGET),
    T.MEMORY_ADDED: ("memory.added", _MEMORY),
    T.MEMORY_DELETED: ("memory.deleted", _MEMORY),
    T.MEMORY_EXPORTED: ("memory.exported", _MEMORY),
    T.SANDBOX_STARTED: (
        "sandbox.started",
        (*_SANDBOX, "code_bytes", "files", "network", "timeout_s"),
    ),
    T.SANDBOX_FINISHED: (
        "sandbox.finished",
        (*_SANDBOX, "exit_code", "timed_out", "duration_ms", "truncated", "files"),
    ),
    T.SANDBOX_FAILED: ("sandbox.failed", (*_SANDBOX, "error_kind", "duration_ms")),
    T.RUN_RECOVERED: ("checkpoint.recovered", ("kind", "step", "replayed", "rerun", "stage")),
}
# Failure kinds of tool and model calls worth an entry of their own.
FAILURES = {
    "guardrail-blocked": "guardrail.blocked",
    "approval-denied": "approval.denied",
    "run-limit-exceeded": "limits.exceeded",
}
APP_PREFIX = "custom.audit."  # ctx.emit("audit.<name>"): recorded as action `app.<name>`
APP_ACTION = "app."  # no first-party action starts with it, so app records never mimic one
_MAX_TRACKED = 10_000  # open approved calls remembered for `approval.consumed`


@dataclass(frozen=True)
class AuditEntry:
    """One security-relevant fact, before it is signed: what, when, which run and call."""

    action: str
    ts: float
    run_id: str = ""
    call_id: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)
    source: str = "sdk"  # "app": from the host's own `ctx.emit("audit.<name>")`

    def body(self) -> dict[str, Any]:
        """The unsigned record of this entry."""
        return {
            "kind": "event",
            "action": self.action,
            "ts": round(self.ts, 6),
            "run_id": self.run_id,
            "call_id": self.call_id,
            "source": self.source,
            "data": dict(self.data),
        }


class Mapper:
    """Turns events into audit entries; remembers approved calls to log their use."""

    def __init__(self, digest: Digest) -> None:
        self._digest = digest
        self._lock = threading.Lock()
        self._decided: OrderedDict[tuple[str, str], tuple[str, bool]] = OrderedDict()

    def entries(self, event: Event) -> list[AuditEntry]:
        """The entries `event` produces (most events produce none)."""
        data = event.data
        if event.type in DIRECT:
            action, keys = DIRECT[event.type]
            return [self._entry(event, action, clean_data(data, keys, self._digest))]
        if event.type == T.APPROVAL_DECIDED:
            return self._decided_entries(event)
        if event.type in (T.TOOL_FINISHED, T.TOOL_FAILED):
            return self._tool_end(event)
        if event.type == T.LLM_FAILED and data.get("error_kind") == "guardrail-blocked":
            kept = clean_data(data, ("provider", "model", "stage", "error_kind"), self._digest)
            return [self._entry(event, "guardrail.blocked", kept)]
        if event.type.startswith(APP_PREFIX) and len(event.type) > len(APP_PREFIX):
            action = APP_ACTION + event.type[len(APP_PREFIX) :]
            kept = clean_data(data, None, self._digest)
            return [self._entry(event, action, kept, source="app")]
        return []

    def _decided_entries(self, event: Event) -> list[AuditEntry]:
        data = event.data
        kept = clean_data(data, ("tool", "approval_id", "approved", "edited"), self._digest)
        if data.get("edited_args") is not None:  # the edit itself never lands, only its digest
            kept["edited"] = True
            kept["edited_digest"] = self._digest(data["edited_args"])
        approved = data.get("approved") is True
        with self._lock:
            key = (event.run_id, event.call_id)
            self._decided[key] = (str(data.get("approval_id", "")), approved)
            while len(self._decided) > _MAX_TRACKED:
                self._decided.popitem(last=False)
        out = [self._entry(event, "approval.decided", kept)]
        if not approved:
            ids = {"tool": kept.get("tool"), "approval_id": kept.get("approval_id")}
            out.append(self._entry(event, "approval.denied", ids))
        return out

    def _tool_end(self, event: Event) -> list[AuditEntry]:
        with self._lock:
            decided = self._decided.pop((event.run_id, event.call_id), None)
        out: list[AuditEntry] = []
        tool = event.data.get("tool")
        if decided is not None and decided[1]:
            ok = event.type == T.TOOL_FINISHED
            used = clean_data(
                {"tool": tool, "approval_id": decided[0], "ok": ok}, None, self._digest
            )
            out.append(self._entry(event, "approval.consumed", used))
        kind = str(event.data.get("error_kind", "")) if event.type == T.TOOL_FAILED else ""
        action = FAILURES.get(kind)
        already_denied = decided is not None and not decided[1] and action == "approval.denied"
        if action is not None and not already_denied:
            kept = clean_data({"tool": tool, "error_kind": kind}, None, self._digest)
            out.append(self._entry(event, action, kept))
        return out

    @staticmethod
    def _entry(
        event: Event, action: str, data: dict[str, Any], *, source: str = "sdk"
    ) -> AuditEntry:
        ids = (clean_value(event.run_id, name=True), clean_value(event.call_id, name=True))
        run, call = str(ids[0] or ""), str(ids[1] or "")
        return AuditEntry(action, event.ts, run, call, data, source)
