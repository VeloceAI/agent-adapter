"""HTTP handlers for `/api/chat/...`; the server checked Host and the token already.

Every POST also passes the Origin and `X-AA-CSRF` checks and needs `Content-Type:
application/json` (body: one JSON object, at most 64 KB and 64 levels deep; an empty body
is `{}`), except `POST /api/chat/files?session=<sid>`, which takes
`application/octet-stream` and the file name in `X-AA-Filename` (percent-encoded UTF-8).
Errors are `{"error": "<why>"}`.

`GET /api/chat/turns/<tid>/stream?since=N` is Server-Sent Events. Each frame is
`id: <seq>`, `event: <type>`, `data: <one JSON line, ASCII>`; a `: ping` comment comes every
5 s without events. Every data object has `seq` (0, 1, ... per turn) and `type`; reconnect
with `since` = last seq + 1. The stream always ends with the turn's one `done` or `error`
event (sent again when `since` is past it):

- `text`: `{"text"}`, a chunk of the answer.
- `tool`: `{"call_id", "name", "status": "running|ok|failed|pending", "ms", "arguments"?,
  "output"?, "error_kind"?, "cached"?}`; the whole summary on every change.
- `approval`: `{"approval_id", "turn_id", "session_id", "call_id", "tool", "arguments",
  "hint", "status": "pending|approved|denied|timed-out|stopped", "reason", "decided_by",
  "created_at"}`.
- `state`: `{"state"}`, the session state (secret-named keys masked), at start and end.
- `usage`: `{"input_tokens", "output_tokens", "cost_usd"}`, the turn's totals so far.
- `memory`: `{"recalled": [record]}` at the start, `{"saved": n, "error"?}` after a save
  (see `memory_routes`); approval events carry `context`, similar past decisions.
- `done`: `{"status": "done", "turn"}`; `error`: `{"status": "failed|stopped|timed-out",
  "kind", "message", "turn"}`. `turn` is the stored turn (see `GET .../sessions/<sid>`).

`arguments` and `output` are JSON text cut to 2 KB with secret-named keys masked.
"""

from __future__ import annotations

import json
import logging
import re
from http import HTTPStatus
from io import BufferedIOBase
from typing import Any, Protocol
from urllib.parse import quote, unquote

from agent_adapter.core.errors import ApprovalAlreadyDecided, ApprovalNotFound
from agent_adapter.hooks.builtin.logging import redact_secrets
from agent_adapter.viewer._drain import drain
from agent_adapter.viewer.chat.files import MAX_FILE
from agent_adapter.viewer.chat.host import ChatHost
from agent_adapter.viewer.chat.memory_routes import (
    DELETE_PATH,
    MEMORY_PATH,
    memory_delete,
    memory_listing,
)
from agent_adapter.viewer.chat.records import Refused
from agent_adapter.viewer.chat.stream import stream_turn

logger = logging.getLogger(__name__)

MAX_JSON = 64 * 1024
MAX_DEPTH = 64  # nesting levels of a JSON body
FILENAME_HEADER = "X-AA-Filename"
_ID = r"[0-9a-f]{16}"
_SESSION = re.compile(rf"^/api/chat/sessions/(ses_{_ID})$")
_SESSION_POST = re.compile(rf"^/api/chat/sessions/(ses_{_ID})/(state|turns)$")
_STREAM = re.compile(rf"^/api/chat/turns/(turn_{_ID})/stream$")
_TURN_POST = re.compile(rf"^/api/chat/turns/(turn_{_ID})/(stop|replay|eval-case)$")
_DECIDE = re.compile(r"^/api/chat/approvals/([A-Za-z0-9_.:-]{1,200})/decide$")
_FILE = re.compile(rf"^/api/chat/files/(fil_{_ID})$")
_TURN_ID = re.compile(rf"^turn_{_ID}$")
_OFF = {"enabled": False, "mode": "", "agent_name": ""}


class Exchange(Protocol):
    """What the routes need of the server's request handler."""

    headers: Any
    rfile: BufferedIOBase
    wfile: BufferedIOBase
    close_connection: bool

    def reply_json(self, status: int, payload: Any) -> None: ...

    def reply_bytes(
        self, status: int, body: bytes, content_type: str, headers: dict[str, str]
    ) -> None: ...

    def refusal(self, content_type: str) -> str: ...

    def begin_stream(self) -> None: ...


def handle_get(ex: Exchange, host: ChatHost | None, path: str, query: dict[str, list[str]]) -> None:
    """Answer a GET under `/api/chat`."""
    if path == "/api/chat":
        ex.reply_json(HTTPStatus.OK, host.info() if host is not None else _OFF)
        return
    if host is None:
        ex.reply_json(HTTPStatus.NOT_FOUND, {"error": "chat is off"})
        return
    try:
        _get(ex, host, path, query)
    except Refused as refused:
        ex.reply_json(refused.status, {"error": str(refused)})


def _get(ex: Exchange, host: ChatHost, path: str, query: dict[str, list[str]]) -> None:
    if path == MEMORY_PATH:
        ex.reply_json(HTTPStatus.OK, memory_listing(host))
    elif path == "/api/chat/sessions":
        ex.reply_json(HTTPStatus.OK, {"sessions": host.store.summaries()})
    elif (match := _SESSION.match(path)) is not None:
        ex.reply_json(HTTPStatus.OK, session_detail(host, match.group(1)))
    elif (match := _STREAM.match(path)) is not None:
        live = host.live(match.group(1))
        stream_turn(ex, live, since=_since(query), heartbeat_s=host.options.heartbeat_s)
    elif (match := _FILE.match(path)) is not None:
        ref, data = host.files.read(match.group(1))
        text = ref.media_type.startswith("text/")
        media = f"{ref.media_type}; charset=utf-8" if text else ref.media_type
        disposition = f"attachment; filename*=UTF-8''{quote(ref.name, safe='')}"
        ex.reply_bytes(HTTPStatus.OK, data, media, {"Content-Disposition": disposition})
    else:
        ex.reply_json(HTTPStatus.NOT_FOUND, {"error": "not found"})


def session_detail(host: ChatHost, session_id: str) -> dict[str, Any]:
    """A session with its turns, state (masked), branches and files."""
    session = host.store.get(session_id)
    with host.store.lock:
        detail = {
            "session": session.summary(),
            "state": redact_secrets(session.saved_state),
            "turns": [turn.view() for turn in session.turns],
            "files": [ref.view() for ref in session.files.values()],
        }
    detail["branches"] = host.store.branches(session_id)
    return detail


def handle_post(
    ex: Exchange, host: ChatHost | None, path: str, query: dict[str, list[str]]
) -> None:
    """Answer a POST under `/api/chat` after the Origin, CSRF and content type checks."""
    upload = path == "/api/chat/files"
    try:
        raw = _read_body(ex, MAX_FILE if upload else MAX_JSON)
    except Refused as refused:
        ex.reply_json(refused.status, {"error": str(refused)})
        return
    if host is None:
        ex.reply_json(HTTPStatus.NOT_FOUND, {"error": "chat is off"})
        return
    refusal = ex.refusal("application/octet-stream" if upload else "application/json")
    if refusal:
        logger.warning("viewer: refused a chat request: %s", refusal)
        ex.reply_json(HTTPStatus.FORBIDDEN, {"error": refusal})
        return
    try:
        if upload:
            _upload(ex, host, query, raw)
        else:
            _post(ex, host, path, _json_object(raw))
    except Refused as refused:
        ex.reply_json(refused.status, {"error": str(refused)})
    except ApprovalNotFound:
        ex.reply_json(HTTPStatus.NOT_FOUND, {"error": "no such approval"})
    except ApprovalAlreadyDecided as exc:
        ex.reply_json(HTTPStatus.CONFLICT, {"error": f"already {exc.status}"})


def _post(ex: Exchange, host: ChatHost, path: str, body: dict[str, Any]) -> None:
    if path == DELETE_PATH:
        ex.reply_json(HTTPStatus.OK, memory_delete(host, body))
    elif path == "/api/chat/sessions":
        title, state = body.get("title", ""), body.get("state", {})
        if not isinstance(title, str) or not isinstance(state, dict):
            raise Refused(HTTPStatus.BAD_REQUEST, 'send {"title"?: text, "state"?: object}')
        session = host.store.create(title, state)
        ex.reply_json(HTTPStatus.CREATED, session_detail(host, session.id))
    elif (match := _SESSION_POST.match(path)) is not None:
        session_id, action = match.groups()
        if action == "state":
            state = host.set_state(session_id, body.get("state"))
            ex.reply_json(HTTPStatus.OK, {"state": state})
        else:
            edit_of = body.get("edit_of") or ""
            if not isinstance(edit_of, str) or (edit_of and not _TURN_ID.match(edit_of)):
                raise Refused(HTTPStatus.BAD_REQUEST, "edit_of must be a turn id")
            started = host.start(
                session_id, body.get("text"), body.get("attachment_ids"), edit_of=edit_of
            )
            ex.reply_json(HTTPStatus.CREATED, started)
    elif (match := _TURN_POST.match(path)) is not None:
        _turn_action(ex, host, match.group(1), match.group(2), body)
    elif (match := _DECIDE.match(path)) is not None:
        approved, reason = body.get("approved"), body.get("reason", "")
        if not isinstance(approved, bool) or not isinstance(reason, str):
            raise Refused(HTTPStatus.BAD_REQUEST, 'send {"approved": true|false, "reason"?: text}')
        view = host.desk.decide(match.group(1), approved=approved, reason=reason)
        ex.reply_json(HTTPStatus.OK, {"approval": view})
    else:
        ex.reply_json(HTTPStatus.NOT_FOUND, {"error": "not found"})


def _turn_action(
    ex: Exchange, host: ChatHost, turn_id: str, action: str, body: dict[str, Any]
) -> None:
    if action == "stop":
        ex.reply_json(HTTPStatus.OK, host.stop(turn_id))
    elif action == "replay":
        ex.reply_json(HTTPStatus.CREATED, host.replay(turn_id))
    else:
        _, turn = host.store.find_turn(turn_id)
        case = host.evals.add(turn, body.get("id"), body.get("expected"), body.get("tags"))
        ex.reply_json(HTTPStatus.CREATED, {"case": case, "file": host.evals.path.name})


def _upload(ex: Exchange, host: ChatHost, query: dict[str, list[str]], data: bytes) -> None:
    session = host.store.get(query.get("session", [""])[0])
    if not data:
        raise Refused(HTTPStatus.BAD_REQUEST, "send the file bytes with a Content-Length")
    name = unquote(str(ex.headers.get(FILENAME_HEADER, "")), errors="replace")
    try:
        ref = host.files.add(session, name, data)
    except OSError as exc:
        logger.error("chat: storing an upload failed (%s)", type(exc).__name__)
        raise Refused(HTTPStatus.INTERNAL_SERVER_ERROR, "the file was not stored") from exc
    ex.reply_json(
        HTTPStatus.CREATED,
        {"attachment_id": ref.id, "name": ref.name, "size": ref.size, "media_type": ref.media_type},
    )


def _read_body(ex: Exchange, cap: int) -> bytes:
    """The whole body, read before any answer so the client never sees a reset connection.

    Raises:
        Refused: a bad Content-Length (400), over `cap` (413), or a body cut short (400).
    """
    length = _length(ex)
    if length > cap:
        if not drain(ex.rfile, length):
            ex.close_connection = True
        raise Refused(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"a request may hold {cap} bytes")
    if length <= 0:
        return b""
    data = ex.rfile.read(length)
    if len(data) != length:
        raise Refused(HTTPStatus.BAD_REQUEST, "the body ended early")
    return data


def _json_object(raw: bytes) -> dict[str, Any]:
    """The body as a JSON object: at most `MAX_DEPTH` deep, all text valid Unicode.

    Raises:
        Refused: not JSON, not an object, too deep, or a lone surrogate in it (400).
    """
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except RecursionError:
        raise Refused(HTTPStatus.BAD_REQUEST, f"JSON may nest {MAX_DEPTH} levels") from None
    except ValueError:
        raise Refused(HTTPStatus.BAD_REQUEST, "the body is not JSON") from None
    if not isinstance(data, dict):
        raise Refused(HTTPStatus.BAD_REQUEST, "the body must be a JSON object")
    if _depth(data) > MAX_DEPTH:
        raise Refused(HTTPStatus.BAD_REQUEST, f"JSON may nest {MAX_DEPTH} levels")
    try:
        json.dumps(data, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate is valid JSON, yet no file can hold it
        raise Refused(HTTPStatus.BAD_REQUEST, "the text is not valid Unicode") from None
    return data


def _depth(value: Any) -> int:
    """How deeply objects and arrays nest in `value` (no recursion)."""
    deepest, stack = 0, [(value, 1)]
    while stack:
        item, level = stack.pop()
        if isinstance(item, dict | list):
            deepest = max(deepest, level)
            children = item.values() if isinstance(item, dict) else item
            stack.extend((child, level + 1) for child in children)
    return deepest


def _length(ex: Exchange) -> int:
    try:
        return int(ex.headers.get("Content-Length", "0") or "0")
    except ValueError:
        raise Refused(HTTPStatus.BAD_REQUEST, "a bad Content-Length") from None


def _since(query: dict[str, list[str]]) -> int:
    try:
        return max(0, int(query.get("since", ["0"])[0]))
    except ValueError:
        return 0
