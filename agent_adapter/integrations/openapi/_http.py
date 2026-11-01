"""Send one imported operation over httpx and turn the response into a tool result."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, ToolArgsInvalid, ToolError
from agent_adapter.exec.pipeline import current_attempt
from agent_adapter.integrations.openapi._decode import ACCEPT_ENCODING, read_body
from agent_adapter.integrations.openapi._schema import Binding, Operation
from agent_adapter.integrations.openapi.auth import Auth
from agent_adapter.runtime.loop_local import LoopLocal

logger = logging.getLogger(__name__)

BODY_LIMIT = 500  # characters of an error response body shown in the ToolError
MAX_RETRY_AFTER = 30.0  # seconds: the longest `Retry-After` wait before a retry
IDEMPOTENT_METHODS = frozenset({"get", "head", "put", "delete"})
CONTEXT_PARAM = "_aa_ctx"  # never a schema property: sanitised names do not start with "_"
_PRINTABLE = re.compile(r"[\x20-\x7e\t]*")  # header and cookie values: no CR/LF, ASCII only
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


class OperationCall:
    """The tool function of one operation: keyword arguments named as the schema properties.

    Requests go only to `base_url`'s scheme, host and port, redirects are never followed,
    and `timeout` bounds the whole request (connect to last body byte). 5xx and ambiguous
    transport errors of a non-idempotent operation become `ToolError` so the executor does
    not repeat them. A `Retry-After` on a retryable 429/503 is waited for (at most
    `MAX_RETRY_AFTER`) only when the executor says another attempt follows
    (`exec.pipeline.current_attempt()`); the executor's own backoff comes after it.
    """

    def __init__(
        self,
        operation: Operation,
        clients: LoopLocal[Any],
        auth: Auth | None,
        *,
        base_url: str,
        timeout: float = 30.0,
        idempotent: bool | None = None,
    ) -> None:
        self.operation = operation
        self.clients = clients
        self.auth = auth
        self.base_url = base_url
        self.timeout = timeout
        self.idempotent = (
            operation.method in IDEMPOTENT_METHODS if idempotent is None else idempotent
        )

    async def __call__(self, **kwargs: Any) -> Any:
        import httpx

        ctx: ToolContext | None = kwargs.pop(CONTEXT_PARAM, None)
        try:
            response, content = await self._exchange(kwargs)
            return _result(self.operation, response, content, retry_5xx=self.idempotent)
        except httpx.HTTPStatusError as exc:
            if _another_attempt_follows(ctx):
                await self._wait_retry_after(exc.response)
            raise

    async def _exchange(self, kwargs: Mapping[str, Any]) -> tuple[Any, bytes]:
        import httpx

        op = self.operation
        method = op.method.upper()
        path, params, headers, body = _request_parts(op, kwargs)
        headers["Accept-Encoding"] = ACCEPT_ENCODING  # only what read_body can bound
        client = self.clients.get()
        request = client.build_request(method, path, params=params, headers=headers, **body)
        _check_origin(request.url, httpx.URL(self.base_url), op)
        if self.auth is not None:  # credentials only once the destination is known to be ours
            auth_headers: dict[str, str] = {}
            auth_params: dict[str, Any] = {}
            await self.auth.apply(auth_headers, auth_params)
            request = client.build_request(
                method,
                path,
                params={**params, **auth_params},
                headers={**headers, **auth_headers},
                **body,
            )
            _check_origin(request.url, httpx.URL(self.base_url), op)
        logger.debug("openapi %s: %s %s", op.name, method, op.path)
        try:
            return await self._send(client, request)
        except httpx.TransportError as exc:
            _redact_request(exc)
            raise self._transport_error(exc) from exc

    async def _send(self, client: Any, request: Any) -> tuple[Any, bytes]:
        import httpx

        op = self.operation
        timer = asyncio.timeout(self.timeout)
        try:
            async with timer:
                # Never follow: a redirect would carry our credentials to another origin.
                response = await client.send(request, stream=True, follow_redirects=False)
                try:
                    where = f"{op.method.upper()} {op.path}"
                    content = await read_body(response, where=where, tool=op.name)
                finally:
                    await response.aclose()
        except TimeoutError as exc:
            if not timer.expired():
                raise
            raise httpx.ReadTimeout(
                f"no complete response within {self.timeout:g}s", request=request
            ) from exc
        return response, content

    def _transport_error(self, exc: Exception) -> Exception:
        """`exc` itself when a retry is safe, else a ToolError the executor does not retry."""
        import httpx

        op = self.operation
        safe = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)  # nothing was sent
        if self.idempotent or isinstance(exc, safe):
            return exc
        logger.warning("openapi %s: %s on a non-idempotent call", op.name, type(exc).__name__)
        return ToolError(
            f"{type(exc).__name__} from {op.method.upper()} {op.path}; not retried because "
            f"{op.method.upper()} is not idempotent",
            tool=op.name,
            details={"error": type(exc).__name__},
        )

    async def _wait_retry_after(self, response: Any) -> None:
        if response.status_code not in (429, 503):
            return
        delay = retry_after(response.headers.get("retry-after"))
        if delay is not None:
            logger.info("openapi %s: waiting %.1fs (Retry-After)", self.operation.name, delay)
            await _sleep(delay)


def retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """Seconds a `Retry-After` header asks for, within `[0, MAX_RETRY_AFTER]`; None if absent."""
    if not value:
        return None
    text = value.strip()
    if text.isdigit():
        seconds = float(text)
    else:
        try:
            moment = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            logger.debug("ignoring an unparsable Retry-After header")
            return None
        seconds = moment.timestamp() - (time.time() if now is None else now)
    return min(max(seconds, 0.0), MAX_RETRY_AFTER)


def _another_attempt_follows(ctx: ToolContext | None) -> bool:
    """True when the executor will call the tool again after a retryable failure."""
    attempt = current_attempt()
    if attempt is None or ctx is None or attempt.call_id != ctx.call_id:
        return False  # not this call's executor attempt (a direct invoke): no retry follows
    return not attempt.last


def _redact_request(exc: Any) -> None:
    """Replace the request on an httpx error by a copy without its query string."""
    try:
        request = exc.request
    except RuntimeError:  # httpx raises when no request was attached: nothing to redact
        return
    exc.request = _safe_request(request)


def _safe_request(request: Any) -> Any:
    """A copy of `request` without its query string (it may hold a QueryAuth secret)."""
    import httpx

    return httpx.Request(request.method, request.url.copy_with(query=None))


def _check_origin(url: Any, base: Any, op: Operation) -> None:
    """The request must go to `base`'s scheme, host and port (a defence beyond import checks)."""
    if (url.scheme, url.host, url.port) != (base.scheme, base.host, base.port):
        logger.error("openapi %s: refused a request to %s://%s", op.name, url.scheme, url.host)
        raise ConfigError(
            f"refused: {op.name} would send a request to {url.scheme}://{url.host}, "
            f"not to {base.scheme}://{base.host}"
        )


def _request_parts(
    op: Operation, kwargs: Mapping[str, Any]
) -> tuple[str, dict[str, Any], dict[str, str], dict[str, Any]]:
    path = op.path
    params: dict[str, Any] = {}
    headers: dict[str, str] = {}
    cookies: list[str] = []
    fields: dict[str, Any] = {}
    whole: list[Any] = []
    for binding in op.bindings:
        if binding.prop not in kwargs:
            continue
        value = kwargs[binding.prop]
        if binding.location == "body":
            fields[binding.wire] = value
        elif binding.location == "body_whole":
            whole.append(value)
        elif binding.location == "path":
            path = path.replace(f"{{{binding.wire}}}", _path_value(op, binding, value))
        elif value is None:
            continue  # an explicit null parameter is simply not sent
        elif binding.location == "query":
            params[binding.wire] = _query_value(binding, value)
        elif binding.location == "header":
            headers[binding.wire] = _header_value(op, binding, value)
        else:
            text = _header_value(op, binding, value)
            cookies.append(f"{binding.wire}={quote(text, safe='')}")
    if cookies:
        headers["Cookie"] = "; ".join(cookies)
    return path, params, headers, _body(op, fields, whole)


def _invalid(op: Operation, binding: Binding, message: str) -> ToolArgsInvalid:
    error = {"loc": (binding.prop,), "msg": message, "type": "value_error"}
    return ToolArgsInvalid(op.name, [error])


def _path_value(op: Operation, binding: Binding, value: Any) -> str:
    """One percent-encoded path segment; `/` and `.` included, so no value changes the path."""
    text = "" if value is None else _text(value)
    if text in ("", ".", ".."):
        raise _invalid(op, binding, "a path value must be non-empty and not '.' or '..'")
    return quote(text, safe="").replace(".", "%2E")


def _header_value(op: Operation, binding: Binding, value: Any) -> str:
    text = _text(value)
    if not _PRINTABLE.fullmatch(text):
        raise _invalid(op, binding, f"a {binding.location} value must be printable ASCII")
    return text


def _body(op: Operation, fields: dict[str, Any], whole: list[Any]) -> dict[str, Any]:
    """`{"json": ...}` for httpx, or `{}` when nothing is sent."""
    if whole:
        return {"json": whole[0]}
    if fields or op.body == "required":
        return {"json": fields}
    return {}


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return ",".join(_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _query_value(binding: Binding, value: Any) -> Any:
    if isinstance(value, list | tuple) and binding.explode:
        return [_text(item) for item in value]  # repeated keys
    return _text(value)


def _result(op: Operation, response: Any, content: bytes, *, retry_5xx: bool) -> Any:
    status = response.status_code
    where = f"{op.method.upper()} {op.path}"
    if status == 429 or (status >= 500 and retry_5xx):
        logger.warning("openapi %s: HTTP %d from %s (retryable)", op.name, status, where)
        raise _status_error(status, where, response, content)
    text = content.decode(response.charset_encoding or "utf-8", errors="replace")
    if status >= 300:
        note = (
            "" if status < 500 else f"; not retried because {op.method.upper()} is not idempotent"
        )
        logger.info("openapi %s: HTTP %d from %s", op.name, status, where)
        raise ToolError(
            f"HTTP {status} from {where}{note}: {text[:BODY_LIMIT]}",
            tool=op.name,
            details={"status": status},
        )
    if status == 204 or not content:
        return None
    if "json" in response.headers.get("content-type", "").lower():
        try:
            return json.loads(content)
        except ValueError:
            logger.debug("openapi %s: invalid JSON body returned as text", op.name)
    return text


def _status_error(status: int, where: str, response: Any, content: bytes) -> Exception:
    """An `httpx.HTTPStatusError` whose message, request and response hold no query string."""
    import httpx

    safe = _safe_request(response.request)
    dropped = ("content-encoding", "content-length", "transfer-encoding")  # body is decoded
    headers = [(k, v) for k, v in response.headers.items() if k.lower() not in dropped]
    copy = httpx.Response(status, headers=headers, content=content, request=safe)
    error: Exception = httpx.HTTPStatusError(
        f"HTTP {status} from {where}", request=safe, response=copy
    )
    return error  # typed here: without the extra, httpx is Any to the type check
