"""The audit package's error: a fail-closed log that cannot record stops the call."""

from __future__ import annotations

from agent_adapter.core.errors import AdapterError


class AuditUnavailable(AdapterError):
    """A fail-closed audit log lost a record (or could not store it in time); no call runs."""

    kind = "audit-unavailable"

    def __init__(self, failures: int, *, waiting: int = 0) -> None:
        self.failures = failures
        self.waiting = waiting
        why = f"{waiting} record(s) not stored in time" if waiting else f"{failures} lost"
        super().__init__(
            f"the audit log could not record ({why}); calls are blocked until it is fixed "
            "and re-attached"
        )
