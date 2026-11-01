"""Atomic file writes (temp file in the same directory, then `os.replace`)."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def write_atomic(path: str | os.PathLike[str], text: str) -> None:
    """Write `text` (UTF-8) to `path` so readers see the old file or the new one, never half."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):  # the temp file may already be gone
            os.unlink(tmp)
        logger.error("could not write %s", target)
        raise
