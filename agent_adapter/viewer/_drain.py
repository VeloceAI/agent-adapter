"""Read and discard a refused request body, so the client sees the answer, not a reset."""

from __future__ import annotations

from typing import Protocol


class _Readable(Protocol):
    def read(self, size: int = ..., /) -> bytes: ...


DRAIN_LIMIT = 1024 * 1024  # bytes: larger bodies are not read; the connection closes instead
_CHUNK = 64 * 1024


def drain(rfile: _Readable, length: int) -> bool:
    """Discard `length` body bytes; False when the body is too large to drain or ends early."""
    if length > DRAIN_LIMIT:
        return False
    left = length
    while left > 0:
        chunk = rfile.read(min(_CHUNK, left))
        if not chunk:
            return False
        left -= len(chunk)
    return True
