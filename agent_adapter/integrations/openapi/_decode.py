"""Read a response body with a hard size cap, decompressing gzip / deflate incrementally."""

from __future__ import annotations

import zlib
from typing import Any

from agent_adapter.core.errors import ToolError

MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # raw and decoded body; larger -> ToolError
ACCEPT_ENCODING = "gzip, deflate, identity"  # the only codings we decode


class _Inflater:
    """zlib with bounded output.

    gzip may hold several members back to back (RFC 1952 §2.2): each ends one
    decompressor and the bytes after it start the next. NUL bytes after the last member are
    padding and ignored, as the gzip tool does; anything else after them is an error. HTTP
    `deflate` may be zlib-wrapped or raw (both are seen) and is one stream: bytes after its
    end are an error.
    """

    def __init__(self, coding: str) -> None:
        self.gzip = coding in ("gzip", "x-gzip")
        self.raw_fallback = not self.gzip
        self.inflater = self._new()
        self.started = False
        self.padding = False  # past the last gzip member: only NUL bytes may follow

    def _new(self) -> zlib._Decompress:
        return zlib.decompressobj(16 + zlib.MAX_WBITS if self.gzip else zlib.MAX_WBITS)

    def feed(self, data: bytes, room: int) -> bytes:
        """At most `room` bytes of output; the rest waits in `tail`.

        Raises:
            zlib.error: corrupt data, or data after the end of a deflate stream.
        """
        if self.inflater.eof:  # a new gzip member, padding, or trailing bytes
            if not self.gzip:
                raise zlib.error("data after the end of the deflate stream")
            if self.padding or data[:1] == b"\0":  # a member starts with 0x1f, never NUL
                if data.strip(b"\0"):
                    raise zlib.error("data after the NUL padding of the gzip body")
                self.padding = True
                return b""
            self.inflater = self._new()
        try:
            out = self.inflater.decompress(data, room)
        except zlib.error:
            if not (self.raw_fallback and not self.started):
                raise
            self.inflater = zlib.decompressobj(-zlib.MAX_WBITS)
            out = self.inflater.decompress(data, room)
        self.started = True
        return out

    @property
    def tail(self) -> bytes:
        """Input still to feed: what did not fit in `room`, or what follows a finished member."""
        if self.padding:
            return b""  # fed whole
        if self.inflater.eof:
            return self.inflater.unused_data
        return self.inflater.unconsumed_tail

    def finish(self) -> bytes:
        """The last output; the stream must have ended (a truncated body is an error)."""
        if not self.started:
            return b""  # no body at all (a 204, a HEAD) despite the Content-Encoding header
        out = self.inflater.flush()
        if not self.inflater.eof:
            raise zlib.error("the compressed body ended early")
        return out


def _inflater(header: str, where: str, tool: str) -> _Inflater | None:
    codings = [c.strip().lower() for c in header.split(",") if c.strip()]
    codings = [c for c in codings if c != "identity"]
    if not codings:
        return None
    if len(codings) == 1 and codings[0] in ("gzip", "x-gzip", "deflate"):
        return _Inflater(codings[0])
    raise ToolError(f"unsupported content encoding {header!r} from {where}", tool=tool)


async def read_body(response: Any, *, where: str, tool: str) -> bytes:
    """The decoded body, never holding more than `MAX_RESPONSE_BYTES` (plus one chunk).

    Raises:
        ToolError: larger than the cap, an unsupported encoding, or undecodable data.
    """
    if response.is_stream_consumed:  # an in-process transport handed back a read response
        if len(response.content) > MAX_RESPONSE_BYTES:
            message = f"the response from {where} is larger than {MAX_RESPONSE_BYTES} bytes"
            raise ToolError(message, tool=tool)
        return bytes(response.content)
    inflater = _inflater(response.headers.get("content-encoding", ""), where, tool)
    body = bytearray()
    raw_size = 0

    def too_large() -> ToolError:
        message = f"the response from {where} is larger than {MAX_RESPONSE_BYTES} bytes"
        return ToolError(message, tool=tool)

    def add(data: bytes) -> None:
        if len(body) + len(data) > MAX_RESPONSE_BYTES:
            raise too_large()
        body.extend(data)

    try:
        async for raw in response.aiter_raw():
            raw_size += len(raw)
            if raw_size > MAX_RESPONSE_BYTES:
                raise too_large()
            if inflater is None:
                add(raw)
                continue
            data = raw
            while data:
                add(inflater.feed(data, MAX_RESPONSE_BYTES - len(body) + 1))
                data = inflater.tail
        if inflater is not None:
            add(inflater.finish())
    except zlib.error as exc:
        raise ToolError(f"the response from {where} could not be decoded", tool=tool) from exc
    return bytes(body)
