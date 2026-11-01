"""Keep `QueryAuth` secrets out of log records (httpx logs every request URL at INFO)."""

from __future__ import annotations

import logging
import re
import threading
from urllib.parse import quote, quote_plus

MASK = "***"
_LOGGERS = ("httpx",)  # httpx's "HTTP Request: GET <url> ..." line

_lock = threading.Lock()
_values: set[str] = set()
_names: set[str] = set()


def protect_query_secret(name: str, value: str) -> None:
    """Mask `value` (and `name=` query values) in records of the httpx logger from now on."""
    with _lock:
        _values.update({value, quote(value, safe=""), quote_plus(value)})
        _names.add(name)
    for logger_name in _LOGGERS:
        target = logging.getLogger(logger_name)
        if _FILTER not in target.filters:
            target.addFilter(_FILTER)


def redact(text: str) -> str:
    """`text` with every protected secret and `name=` query value masked."""
    with _lock:
        values = sorted(_values, key=len, reverse=True)
        names = list(_names)
    for value in values:
        text = text.replace(value, MASK)
    for name in names:
        pattern = rf"([?&](?:{re.escape(name)}|{re.escape(quote_plus(name))})=)[^&#\s\"']*"
        text = re.sub(pattern, rf"\1{MASK}", text)
    return text


class _RedactFilter(logging.Filter):
    """Rewrites a record's message when it holds a protected secret; never drops records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # a malformed record is the logging module's problem, not ours
            return True
        masked = redact(message)
        if masked != message:
            record.msg, record.args = masked, None
        return True


_FILTER = _RedactFilter()
