"""RFC 9457 problem details: one body shape for every error the server returns."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from agent_adapter.core.errors import AdapterError
from agent_adapter.core.types import ErrorInfo

PROBLEM_MEDIA_TYPE = "application/problem+json"
TYPE_PREFIX = "urn:agent-adapter:error:"

# kind -> (status, title); kinds outside the error catalogue are the server's own route problems.
_KINDS: dict[str, tuple[int, str]] = {
    "tool-not-found": (404, "Tool not found"),
    "approval-not-found": (404, "Approval not found"),
    "not-found": (404, "Not found"),
    "method-not-allowed": (405, "Method not allowed"),
    "run-limit-exceeded": (409, "Run limit exceeded"),
    "approval-already-decided": (409, "Approval already decided"),
    "request-in-progress": (409, "Request in progress"),
    "payload-too-large": (413, "Payload too large"),
    "unsupported-media-type": (415, "Unsupported media type"),
    "misdirected-request": (421, "Misdirected request"),
    "request-timeout": (408, "Request timeout"),
    "tool-args-invalid": (422, "Invalid tool arguments"),
    "budget-exceeded": (429, "Budget exceeded"),
    "too-many-streams": (429, "Too many event streams"),
    "service-unavailable": (503, "Service unavailable"),
    "bad-request": (400, "Bad request"),
    "tool-usage-error": (400, "Bad request"),
    "tool-timeout": (504, "Tool timed out"),
    "unauthorized": (401, "Unauthorized"),
    "tool-failed": (502, "Tool failed"),
    "tool-error": (502, "Tool error"),
    "forbidden": (403, "Forbidden"),
    "non-local-client": (403, "Non-local client"),
    "approval-denied": (403, "Approval denied"),
    "guardrail-blocked": (403, "Blocked by a guardrail"),
    "tool-definition-invalid": (500, "Tool definition invalid"),
    "schema-not-portable": (500, "Schema not portable"),
    "config-invalid": (500, "Server misconfigured"),
    "internal-error": (500, "Internal error"),
}
_MODEL = (502, "Model call failed")
_OTHER = (500, "Internal error")


class Problem(BaseModel):
    """An RFC 9457 problem; extensions (`tool`, `errors`, `approval_id`, ...) are extra fields."""

    model_config = ConfigDict(extra="allow", frozen=True)

    type: str
    title: str
    status: int
    detail: str = ""


def status_for(kind: str) -> int:
    """The HTTP status of an error kind."""
    return _entry(kind)[0]


def problem(kind: str, detail: str = "", **extensions: Any) -> Problem:
    """A problem of `kind` with its status and title; `detail` must hold no secrets."""
    status, title = _entry(kind)
    return Problem(type=TYPE_PREFIX + kind, title=title, status=status, detail=detail, **extensions)


def problem_for(exc: BaseException) -> Problem:
    """The problem for an exception: `detail` is `ErrorInfo.message` (the class name only
    for exceptions that are not `AdapterError`), plus `tool`, `errors`, `approval_id`."""
    info = ErrorInfo.from_exception(exc)
    extensions: dict[str, Any] = {}
    if isinstance(exc, AdapterError):
        tool = getattr(exc, "tool", None)
        if isinstance(tool, str) and tool:
            extensions["tool"] = tool
        approval_id = getattr(exc, "approval_id", None)
        if isinstance(approval_id, str) and approval_id:
            extensions["approval_id"] = approval_id
        if info.kind == "tool-args-invalid":
            extensions["errors"] = info.details.get("errors", [])
    return problem(info.kind, info.message, **extensions)


def _entry(kind: str) -> tuple[int, str]:
    if kind in _KINDS:
        return _KINDS[kind]
    return _MODEL if kind.startswith("model-") else _OTHER
