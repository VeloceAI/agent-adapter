"""The local viewer server: static page, run data and the approval desk, on 127.0.0.1 only."""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import secrets
import socket
import sqlite3
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

from agent_adapter.core.errors import (
    AdapterError,
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ConfigError,
    ToolUsageError,
    log_failure,
)
from agent_adapter.evals.store import DEFAULT_EVALS_FOLDER
from agent_adapter.viewer._case_routes import CASE_PATH, case_request
from agent_adapter.viewer._drain import drain
from agent_adapter.viewer._evals import EvalFolder, evals_route
from agent_adapter.viewer._runs_routes import TAGS_PATH, SearchCache, runs_listing, tags_request
from agent_adapter.viewer._tags import TagStore
from agent_adapter.viewer.approvals import ApprovalDesk
from agent_adapter.viewer.assets import page, static_file
from agent_adapter.viewer.chat.evalcases import EvalCases
from agent_adapter.viewer.chat.routes import handle_get, handle_post
from agent_adapter.viewer.chat.setup import DEFAULT_CHAT_DIR
from agent_adapter.viewer.chat.stream import CLIENT_GONE
from agent_adapter.viewer.launch import PATH as LAUNCH_PATH
from agent_adapter.viewer.launch import SCRUB, LaunchCodes
from agent_adapter.viewer.runs import RunFolder

if TYPE_CHECKING:
    from agent_adapter.viewer.chat.host import ChatHost

_Route = Callable[[Any, "ChatHost | None", str, dict[str, list[str]]], None]

logger = logging.getLogger(__name__)

LOOPBACK = "127.0.0.1"
DEFAULT_PORT = 8780
CSRF_HEADER = "X-AA-CSRF"
TOKEN_HEADER = "X-AA-Token"
MAX_BODY = 4_096
_RUN = re.compile(r"^/api/runs/([A-Za-z0-9_-]{1,64})$")
_DECIDE = re.compile(r"^/api/approvals/([A-Za-z0-9_.:-]{1,200})/decide$")
_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "X-Frame-Options": "DENY",
}


def check_host(host: str) -> None:
    """Refuse any bind address but the IPv4 loopback.

    Raises:
        ConfigError: `host` is not 127.0.0.1.
    """
    if host != LOOPBACK:
        raise ConfigError(
            f"the viewer binds only to {LOOPBACK} (got {host!r}): it shows run data and can "
            "decide approvals, so it is never exposed on a network"
        )


class ViewerServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer holding the runs folder, the approval desk and its secrets.

    `token` travels in the URL fragment of `url` (never sent to a server, a log or a Referer);
    the page keeps it in sessionStorage and sends it as `X-AA-Token` on every `/api/` call.
    `csrf` guards decisions on top. Both are new on every start. The HTML shell and its static
    files hold no secret and are public.
    """

    daemon_threads = True
    # Windows SO_REUSEADDR lets a second socket bind a busy port; take it exclusively there.
    allow_reuse_address = os.name != "nt"

    def __init__(
        self,
        folder: RunFolder,
        desk: ApprovalDesk | None,
        port: int,
        chat: ChatHost | None = None,
        evals: EvalFolder | None = None,
        cases_dir: Path | None = None,
    ) -> None:
        self.folder = folder
        self.tags = TagStore(folder.tags_folder)
        self.search = SearchCache()
        self.evals = evals if evals is not None else EvalFolder()
        self.desk = desk
        self.chat = chat
        home = cases_dir if cases_dir is not None else Path(DEFAULT_CHAT_DIR)
        self.cases = chat.evals if chat is not None else EvalCases(home)
        self.token = secrets.token_urlsafe(32)
        self.csrf = secrets.token_urlsafe(32)
        self.refusals = 0  # requests without the token; only the first is a WARNING
        self.launch = LaunchCodes()
        super().__init__((LOOPBACK, port), _Handler)

    def server_bind(self) -> None:
        if os.name == "nt":
            exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if exclusive is not None:
                self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()

    def server_close(self) -> None:
        """Close the socket; with chat on, stop running turns and the chat loop first."""
        if self.chat is not None:
            self.chat.close()
        super().server_close()

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK}:{self.port}/"

    @property
    def url(self) -> str:
        """The address to open: it carries the access token, so keep it private."""
        return f"{self.base_url}#token={self.token}"

    def trusted_hosts(self) -> set[str]:
        """Host header values that name this server (DNS-rebinding guard)."""
        return {f"{LOOPBACK}:{self.port}", f"localhost:{self.port}"}


def make_server(
    target: str | Path,
    *,
    host: str = LOOPBACK,
    port: int = DEFAULT_PORT,
    approvals: str | Path | None = None,
    reviewer: str = "",
    chat: ChatHost | None = None,
    evals: str | Path = DEFAULT_EVALS_FOLDER,
    cases_dir: str | Path = DEFAULT_CHAT_DIR,
) -> ViewerServer:
    """A bound, not yet serving, viewer server (`port=0` picks a free port); `chat` turns chat on.

    `evals` is the folder of saved eval reports the Evals view lists; runs saved as eval
    cases go to the chat's `eval_cases.jsonl`, or without chat to the one in `cases_dir`.

    Raises:
        ConfigError: a non-loopback `host`, a bad target, or a bad approvals database/reviewer.
        OSError: the port is taken.
    """
    check_host(host)
    folder = RunFolder(target)
    desk = ApprovalDesk(approvals, reviewer) if approvals is not None else None
    server = ViewerServer(folder, desk, port, chat, EvalFolder(evals), Path(cases_dir))
    logger.info("viewer on %s for %s", server.base_url, folder.target)
    return server


class _Handler(BaseHTTPRequestHandler):
    server: ViewerServer
    server_version = "agent-adapter-viewer"
    sys_version = ""
    timeout = 10  # seconds; a slow or idle client never pins a thread

    def do_GET(self) -> None:
        if not self._host_ok() or not self._access_ok():
            return
        parts = urlsplit(self.path)
        path = parts.path
        if (launch := LAUNCH_PATH.match(path)) is not None:
            self._launch(launch.group(1))
            return
        if _is_chat(path):
            self._chat(handle_get, path, parse_qs(parts.query))
            return
        try:
            self._get(path, parse_qs(parts.query))
        except (OSError, sqlite3.Error, AdapterError) as exc:
            log_failure(logger, logging.ERROR, exc, "viewer: GET %s failed", path)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "the viewer could not read it"})

    def _get(self, path: str, query: dict[str, list[str]]) -> None:
        chat = self.server.chat
        if path in ("/", "/index.html"):
            mode = chat.mode if chat is not None else ""
            body = page(self.server.csrf, approvals=self.server.desk is not None, chat=mode)
            self._send(HTTPStatus.OK, body.encode("utf-8"), "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            found = static_file(path.removeprefix("/static/"))
            if found is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            else:
                self._send(HTTPStatus.OK, *found)
        elif path == "/api/runs":
            self._json(
                HTTPStatus.OK,
                runs_listing(self.server.folder, self.server.tags, query, self.server.search),
            )
        elif (match := _RUN.match(path)) is not None:
            self.server.folder.refresh()
            detail = self.server.folder.detail(match.group(1), since=_since(query))
            if detail is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "no such run"})
            else:
                self._json(HTTPStatus.OK, detail)
        elif path == "/api/evals" or path.startswith("/api/evals/"):
            self._json(*evals_route(self.server.evals, path, query))
        elif path == "/api/approvals" and self.server.desk is not None:
            self._json(HTTPStatus.OK, {"approvals": self.server.desk.pending()})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._host_ok() or not self._access_ok():
            return
        parts = urlsplit(self.path)
        if _is_chat(parts.path):
            self._chat(handle_post, parts.path, parse_qs(parts.query))
            return
        if (tagged := TAGS_PATH.match(parts.path)) is not None:
            tags_request(self, self.server.folder, self.server.tags, tagged.group(1))
            return
        if (asked := CASE_PATH.match(parts.path)) is not None:
            server = self.server
            case_request(self, server.folder, server.tags, server.cases, asked.group(1))
            return
        match = _DECIDE.match(self.path)
        desk = self.server.desk
        if match is None or desk is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        refusal = self._post_refusal()
        if refusal:
            logger.warning("viewer: refused a decide request: %s", refusal)
            self._json(HTTPStatus.FORBIDDEN, {"error": refusal})
            return
        body = self._body()
        if body is None:
            self._json(HTTPStatus.BAD_REQUEST, {"error": 'send JSON {"approved": true|false}'})
            return
        try:
            record = desk.decide(match.group(1), approved=body[0], reason=body[1])
        except ApprovalNotFound:
            self._json(HTTPStatus.NOT_FOUND, {"error": "no such approval"})
        except ApprovalAlreadyDecided as exc:
            self._json(HTTPStatus.CONFLICT, {"error": f"already {exc.status}"})
        except ToolUsageError as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except (AdapterError, sqlite3.Error, OSError) as exc:
            log_failure(logger, logging.ERROR, exc, "viewer: decide %s failed", match.group(1))
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "the decision was not stored"})
        else:
            self._json(HTTPStatus.OK, {"approval": record})

    def do_PUT(self) -> None:
        self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method not allowed"})

    do_DELETE = do_PATCH = do_PUT

    def _host_ok(self) -> bool:
        if self.headers.get("Host", "") in self.server.trusted_hosts():
            return True
        logger.warning("viewer: refused a request for host %r", self.headers.get("Host", ""))
        self._json(HTTPStatus.FORBIDDEN, {"error": "unknown Host header"})
        return False

    def _access_ok(self) -> bool:
        """The page shell is public; every `/api/` request needs the `X-AA-Token` header."""
        if not urlsplit(self.path).path.startswith("/api/"):
            return True
        given = self.headers.get(TOKEN_HEADER, "")
        if hmac.compare_digest(given.encode(), self.server.token.encode()):
            return True
        self.server.refusals += 1
        # A page left open after a restart polls every second: warn once, then DEBUG.
        level = logging.WARNING if self.server.refusals == 1 else logging.DEBUG
        logger.log(
            level, "viewer: refused %d request(s) without the access token", self.server.refusals
        )
        self._json(
            HTTPStatus.UNAUTHORIZED,
            {"error": "open the address agent-adapter ui printed (it holds the access token)"},
        )
        return False

    def _post_refusal(self, content_type: str = "application/json") -> str:
        """Why a state-changing request is refused: Origin, CSRF token or content type."""
        origin = self.headers.get("Origin", "")
        if origin not in {f"http://{host}" for host in self.server.trusted_hosts()}:
            return "Origin must be this viewer"
        token = self.headers.get(CSRF_HEADER, "")
        if not hmac.compare_digest(token.encode(), self.server.csrf.encode()):
            return "missing or wrong CSRF token"
        if self.headers.get_content_type() != content_type:
            return f"Content-Type must be {content_type}"
        return ""

    def _launch(self, code: str) -> None:
        """A one-use launch code: 302 to the tokened page link, else 404."""
        if not self.server.launch.take(code):
            self._json(HTTPStatus.NOT_FOUND, {"error": "this launch link was used or expired"})
            return
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", f"/#token={self.server.token}")
        self.send_header("Content-Length", "0")
        for name, value in _HEADERS.items():
            self.send_header(name, value)
        self.end_headers()

    def _chat(self, route: _Route, path: str, query: dict[str, list[str]]) -> None:
        """Run a chat route; any failure is logged by type and answered with a JSON 500."""
        self._streaming = False
        try:
            route(self, self.server.chat, path, query)
        except CLIENT_GONE as exc:  # the page went away mid-reply: nobody to answer
            logger.debug("viewer: %s %s: client gone (%s)", self.command, path, type(exc).__name__)
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "viewer: %s %s failed", self.command, path)
            if not self._streaming:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "the request failed"})

    # The chat routes' Exchange: replies, the POST checks and the event-stream head.
    def reply_json(self, status: int, payload: Any) -> None:
        # ASCII escapes: agent or tool text with a lone surrogate still encodes.
        body = json.dumps(payload, ensure_ascii=True, default=str).encode("ascii")
        self._send(HTTPStatus(status), body, "application/json; charset=utf-8")

    def reply_bytes(
        self, status: int, body: bytes, content_type: str, headers: dict[str, str]
    ) -> None:
        self._send(HTTPStatus(status), body, content_type, headers)

    def refusal(self, content_type: str) -> str:
        return self._post_refusal(content_type)

    def begin_stream(self) -> None:
        self._streaming = True
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        for name, value in _HEADERS.items():
            self.send_header(name, value)
        self.end_headers()

    def raw_body(self, limit: int = MAX_BODY) -> bytes | None:
        """The request body, or None when it is empty or over `limit` bytes."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length <= 0:
            return None
        if length > limit:
            if not drain(self.rfile, length):
                self.close_connection = True
            return None
        return self.rfile.read(length)

    def _body(self) -> tuple[bool, str] | None:
        raw = self.raw_body()
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(data, dict) or not isinstance(data.get("approved"), bool):
            return None
        reason = data.get("reason", "")
        return data["approved"], reason if isinstance(reason, str) else ""

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in {**_HEADERS, **(headers or {})}.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        text = (format % args).replace(self.server.token, "***")  # the token is a password
        text = SCRUB.sub("/launch/***", text)
        logger.debug("viewer %s %s", self.address_string(), text)


def _is_chat(path: str) -> bool:
    return path == "/api/chat" or path.startswith("/api/chat/")


def _since(query: dict[str, list[str]]) -> int:
    """The `since` event index of a detail request; 0 when absent or not a number."""
    try:
        return max(0, int(query.get("since", ["0"])[0]))
    except ValueError:
        return 0
