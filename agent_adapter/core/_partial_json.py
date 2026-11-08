"""A lenient parser for JSON text that is still streaming in, and a throttled tracker of it."""

from __future__ import annotations

import json
from typing import Any

MAX_DEPTH = 64
MAX_TEXT = 256 * 1024  # characters a tracker keeps; past it, no more partial values
_EVERY_CHUNK = 1024  # below this many characters a tracker parses on every chunk
_LITERALS = {"true": True, "false": False, "null": None}
_NUMBER = set("+-0123456789.eE")
_WS = " \t\r\n"


class _Incomplete(Exception):
    """The text ended inside a token that has no usable prefix."""


def parse_partial_json(text: str) -> tuple[bool, Any]:
    """`(True, value)` for the JSON value the text starts, as far as it goes, else `(False, None)`.

    Open strings keep their prefix; an object member without a complete key and value
    start, and an unfinished number or literal, are left out.
    """
    parser = _Parser(text)
    try:
        parser.skip()
        if parser.pos >= len(text):
            return False, None
        _, value = parser.value(0)
    except (_Incomplete, ValueError, RecursionError):
        return False, None
    return True, value


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def skip(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in _WS:
            self.pos += 1

    def at_end(self) -> bool:
        self.skip()
        return self.pos >= len(self.text)

    def value(self, depth: int) -> tuple[bool, Any]:
        """`(complete, value)`; raises `_Incomplete` when nothing usable was read."""
        if depth > MAX_DEPTH:
            raise ValueError("too deep")
        self.skip()
        if self.pos >= len(self.text):
            raise _Incomplete
        char = self.text[self.pos]
        if char == "{":
            return self.object(depth)
        if char == "[":
            return self.array(depth)
        if char == '"':
            return self.string()
        return True, self.scalar()

    def object(self, depth: int) -> tuple[bool, Any]:
        self.pos += 1
        out: dict[str, Any] = {}
        while True:
            if self.at_end():
                return False, out
            if self.text[self.pos] == "}":
                self.pos += 1
                return True, out
            if self.text[self.pos] == ",":
                self.pos += 1
                continue
            if self.text[self.pos] != '"':
                raise ValueError("expected a key")
            closed, key = self.string()
            if not closed or self.at_end():
                return False, out
            if self.text[self.pos] != ":":
                raise ValueError("expected ':'")
            self.pos += 1
            try:
                complete, item = self.value(depth + 1)
            except _Incomplete:
                return False, out
            out[key] = item
            if not complete:
                return False, out

    def array(self, depth: int) -> tuple[bool, Any]:
        self.pos += 1
        out: list[Any] = []
        while True:
            if self.at_end():
                return False, out
            if self.text[self.pos] == "]":
                self.pos += 1
                return True, out
            if self.text[self.pos] == ",":
                self.pos += 1
                continue
            try:
                complete, item = self.value(depth + 1)
            except _Incomplete:
                return False, out
            out.append(item)
            if not complete:
                return False, out

    def string(self) -> tuple[bool, str]:
        start = self.pos
        self.pos += 1
        text = self.text
        while self.pos < len(text):
            char = text[self.pos]
            if char == "\\":
                self.pos += 2
                continue
            if char == '"':
                self.pos += 1
                return True, str(json.loads(text[start : self.pos]))
            self.pos += 1
        return False, _open_string(text[start + 1 :])

    def scalar(self) -> Any:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ",]}" + _WS:
            self.pos += 1
        token = self.text[start : self.pos]
        if self.pos >= len(self.text):
            raise _Incomplete  # a number or literal may still grow
        if token in _LITERALS:
            return _LITERALS[token]
        if token and set(token) <= _NUMBER:
            return json.loads(token)
        raise ValueError("not JSON")


def _open_string(body: str) -> str:
    """The prefix of an unterminated JSON string body, without a cut escape."""
    cut = _last_escape(body)
    if cut != -1 and not _escape_complete(body[cut:]):
        body = body[:cut]
    try:
        return str(json.loads(f'"{body}"'))
    except ValueError:
        return ""


def _last_escape(body: str) -> int:
    """Where the last escape starts (a backslash that is not itself escaped), or -1."""
    index = -1
    pos = 0
    while (found := body.find("\\", pos)) != -1:
        index, pos = found, found + 2  # the next character belongs to this escape
    return index


def _escape_complete(tail: str) -> bool:
    if len(tail) < 2:
        return False
    return tail[1] != "u" or len(tail) >= 6


class PartialJSON:
    """Collects streamed JSON text; `feed` returns the parsed value when it changed."""

    def __init__(self) -> None:
        self.text = ""
        self._parsed_at = 0
        self._last: tuple[bool, Any] = (False, None)

    def feed(self, chunk: str) -> tuple[bool, Any]:
        """Add `chunk`; `(True, value)` when the value parsed so far changed, else `(False, None)`.

        Past 1 KB, parsing waits for 1/8 more text, so a long output stays linear.
        """
        if len(self.text) >= MAX_TEXT:
            return False, None
        self.text += chunk[: MAX_TEXT - len(self.text)]
        grown = len(self.text) - self._parsed_at
        step = 1 if len(self.text) < _EVERY_CHUNK else self._parsed_at // 8
        if grown < step:
            return False, None
        self._parsed_at = len(self.text)
        found = parse_partial_json(self.text)
        if not found[0] or found == self._last:
            return False, None
        self._last = found
        return found
