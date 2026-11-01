"""One-use launch codes: a browser opens `/launch/<code>` and gets the tokened page link."""

from __future__ import annotations

import hmac
import logging
import re
import secrets
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

LAUNCH_TTL_S = 60.0
PATH = re.compile(r"^/launch/([A-Za-z0-9_-]{1,128})$")
SCRUB = re.compile(r"/launch/[A-Za-z0-9_-]+")  # codes never reach a log


class LaunchCodes:
    """Codes valid once and for `ttl` seconds; a code on a command line is dead after use.

    A code that was issued and is then reused or expired logs a WARNING every time: someone
    else may have opened it. Unknown codes (probes) warn once per start, then log at DEBUG.
    """

    def __init__(
        self, ttl: float = LAUNCH_TTL_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._codes: dict[str, float] = {}  # live code -> expiry
        self._spent: dict[str, float] = {}  # used or expired code -> when it is forgotten
        self.refusals = 0  # unknown codes

    def new(self) -> str:
        """A fresh code."""
        code = secrets.token_urlsafe(32)
        with self._lock:
            self._codes[code] = self._clock() + self.ttl
        return code

    def take(self, code: str) -> bool:
        """True once for a live code; a used, expired or unknown code is refused."""
        with self._lock:
            self._prune(self._clock())
            if (live := _find(self._codes, code)) is not None:
                self._spent[live] = self._codes.pop(live) + self.ttl
                return True
            issued = _find(self._spent, code) is not None
            if not issued:
                self.refusals += 1
            first = self.refusals == 1
        if issued:
            logger.warning(
                "viewer: a launch link was opened again or after it expired; if you did not "
                "do this, someone else may have opened it: restart the viewer"
            )
        else:
            level = logging.WARNING if first else logging.DEBUG
            logger.log(level, "viewer: refused an unknown launch link")
        return False

    def _prune(self, now: float) -> None:
        """Move expired codes to the spent set; forget spent codes after another `ttl`."""
        for code, until in list(self._codes.items()):
            if until <= now:
                self._spent[code] = self._codes.pop(code) + self.ttl
        self._spent = {code: until for code, until in self._spent.items() if until > now}


def _find(codes: dict[str, float], code: str) -> str | None:
    return next((key for key in codes if hmac.compare_digest(key, code)), None)
