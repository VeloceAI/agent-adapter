"""`POST /api/runs/KEY/eval-case`: save a recorded run as an eval case in `eval_cases.jsonl`."""

from __future__ import annotations

import json
import logging
import re
from http import HTTPStatus
from typing import Any, Protocol

from agent_adapter.core.errors import log_failure
from agent_adapter.evals.from_runs import ExpectedFrom, RecordedRun, RunCase, case_from_run
from agent_adapter.viewer._runs_routes import Reply
from agent_adapter.viewer._tags import TagStore
from agent_adapter.viewer.chat.evalcases import EvalCases, check_id, check_tags
from agent_adapter.viewer.chat.records import Refused
from agent_adapter.viewer.runs import RunFolder

MAX_BODY = 256 * 1024  # bytes: room for a hand-written expected value
CASE_PATH = re.compile(r"^/api/runs/([A-Za-z0-9_-]{1,64})/eval-case$")
REGRESSION = "regression"
_FIELDS = frozenset({"id", "expected", "expected_from", "tags", "regression"})
_BODY_HINT = (
    'send JSON {"id"?: text, "expected_from"?: "output"|"none", "expected"?: value, '
    '"tags"?: [names], "regression"?: true}'
)

logger = logging.getLogger(__name__)


class CaseExchange(Protocol):
    """What the eval-case route needs from the HTTP handler."""

    def refusal(self, content_type: str) -> str: ...

    def raw_body(self, limit: int = ...) -> bytes | None: ...

    def reply_json(self, status: int, payload: Any) -> None: ...


def case_request(
    ex: CaseExchange,
    folder: RunFolder,
    tags: TagStore,
    cases: EvalCases,
    key: str,
) -> None:
    """Origin, CSRF and JSON checked first, like every POST; then the case is appended."""
    refusal = ex.refusal("application/json")
    if refusal:
        logger.warning("viewer: refused an eval-case request: %s", refusal)
        ex.reply_json(HTTPStatus.FORBIDDEN, {"error": refusal})
        return
    try:
        ex.reply_json(*save_run_case(folder, tags, cases, key, ex.raw_body(MAX_BODY)))
    except Exception as exc:  # any failure is a JSON 500, never a traceback to the client
        log_failure(logger, logging.ERROR, exc, "viewer: saving run %s as a case failed", key)
        ex.reply_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "the case was not saved"})


def save_run_case(
    folder: RunFolder,
    tags: TagStore,
    cases: EvalCases,
    key: str,
    body: bytes | None,
) -> Reply:
    """Append run `key` as a case; 422 with the reason when the run kept no input.

    `regression: true` saves a failed run with no expected value and the `regression` tag.

    Raises:
        OSError: the runs folder cannot be listed.
    """
    request = _request(body)
    if request is None:
        return HTTPStatus.BAD_REQUEST, {"error": _BODY_HINT}
    folder.refresh()
    detail = folder.detail(key, paged=False)
    if detail is None:
        return HTTPStatus.NOT_FOUND, {"error": "no such run"}
    regression = request.get("regression") is True
    try:
        extra = check_tags(request.get("tags"))
        case_id = check_id(request.get("id") or f"{REGRESSION if regression else 'run'}-{key}")
    except Refused as refused:
        return HTTPStatus(refused.status), {"error": str(refused)}
    asked = "none" if regression else request.get("expected_from", "output")
    if asked not in ("output", "none"):
        return HTTPStatus.BAD_REQUEST, {"error": 'expected_from must be "output" or "none"'}
    expected_from: ExpectedFrom = "none" if asked == "none" else "output"
    made = case_from_run(
        _recorded(key, detail, tags),
        expected_from=expected_from,
        case_id=case_id,
        extra_tags=[*extra, *([REGRESSION] if regression else [])],
    )
    if not isinstance(made, RunCase):
        return HTTPStatus.UNPROCESSABLE_ENTITY, {"error": f"not saved: {made.reason}"}
    record = made.record()
    if "expected" in request and not regression:
        record["expected"] = request["expected"]
    try:
        cases.append(record)
    except Refused as refused:
        return HTTPStatus(refused.status), {"error": str(refused)}
    logger.info("viewer: saved run %s as an eval case", key)
    return HTTPStatus.CREATED, {"case": record, "file": cases.path.name}


def _request(body: bytes | None) -> dict[str, Any] | None:
    if not body:
        return None
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or not set(data) <= _FIELDS:
        return None
    return data


def _recorded(key: str, detail: dict[str, Any], tags: TagStore) -> RecordedRun:
    summary = detail.get("summary") or {}
    metas: list[dict[str, Any]] = detail.get("meta") or []
    return RecordedRun(
        key=key,
        run_id=str(summary.get("run_id") or key),
        status=str(summary.get("status") or "unfinished"),
        capture_content=any(meta.get("capture_content") is True for meta in metas),
        events=tuple(detail.get("events") or ()),
        tags=tuple(tags.get(key)),
    )
