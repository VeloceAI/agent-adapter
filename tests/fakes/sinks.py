"""In-memory EventSink and LedgerPort fakes that record everything."""

from __future__ import annotations

from agent_adapter.core.types import Event, LedgerEntry


class RecordingSink:
    """EventSink keeping every event in `events`."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        """Event types in emission order."""
        return [event.type for event in self.events]

    def of_type(self, prefix: str) -> list[Event]:
        """Events whose type starts with `prefix`."""
        return [event for event in self.events if event.type.startswith(prefix)]


class RecordingLedger:
    """LedgerPort keeping entries in memory; idempotent on (run_id, entry_id)."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], LedgerEntry] = {}
        self.records = 0

    def record(self, entry: LedgerEntry) -> None:
        self.records += 1
        self._entries.setdefault((entry.run_id, entry.entry_id), entry)

    def total_usd(self, run_id: str | None = None, *, thread_id: str | None = None) -> float:
        return sum(entry.cost_usd or 0.0 for entry in self.entries(run_id, thread_id=thread_id))

    def entries(
        self, run_id: str | None = None, *, thread_id: str | None = None
    ) -> list[LedgerEntry]:
        return [
            entry
            for entry in self._entries.values()
            if (run_id is None or entry.run_id == run_id)
            and (thread_id is None or entry.thread_id == thread_id)
        ]
