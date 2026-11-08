"""Policy errors: a policy that fails to load, and a call a policy refuses."""

from __future__ import annotations

from agent_adapter.core.errors import ApprovalDenied, ConfigError


class PolicyError(ConfigError):
    """A policy is malformed, unknown, or would relax a code default; nothing is applied."""

    kind = "policy-invalid"

    def __init__(self, message: str, *, source: str = "", key: str = "", line: int = 0) -> None:
        self.source = source
        self.key = key
        self.line = line
        where = source + (f":{line}" if line else "")
        place = ", ".join(part for part in (where, key and f"at {key}") if part)
        super().__init__(f"{place}: {message}" if place else message)


class PolicyDenied(ApprovalDenied):
    """A policy does not allow this tool here; a denial like any other (`approval-denied`)."""

    def __init__(self, tool: str, *, call_id: str = "", reason: str = "") -> None:
        super().__init__(tool, call_id=call_id, reason=f"policy: {reason}" if reason else "policy")
