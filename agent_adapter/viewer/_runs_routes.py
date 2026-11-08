"""The runs list with search and tags (`GET /api/runs`), and setting tags (`POST .../tags`)."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable
from http import HTTPStatus
from typing import Any, Protocol

from agent_adapter.core.errors import ToolUsageError, log_failure
from agent_adapter.viewer._search import Query, matches, parse_query
from agent_adapter.viewer._tags import MAX_TAGS, TagStore
from agent_adapter.viewer.runs import RunFolder

TAGS_PATH = re.compile(r"^/api/runs/([A-Za-z0-9_-]{1,64})/tags$")
_BODY_HINT = 'send JSON {"add": ["tag"], "remove": ["tag"]}'

logger = logging.getLogger(__name__)

Reply = tuple[HTTPStatus, dict[str, Any]]


class SearchCache:
    """The last search's matching run keys, reused while no run, tag or query changed."""

    def __init__(self) -> None:
        self._last: tuple[Any, set[str]] | None = None
        self._lock = threading.Lock()

    def keys(self, signature: Any, search: Callable[[], set[str]]) -> set[str]:
        """The keys for `signature`: cached, or found by `search` and kept."""
        with self._lock:
            if self._last is not None and self._last[0] == signature:
                return self._last[1]
        found = search()
        with self._lock:
            self._last = (signature, found)
        return found


def runs_listing(
    folder: RunFolder,
    tags: TagStore,
    query: dict[str, list[str]],
    cache: SearchCache | None = None,
) -> dict[str, Any]:
    """Run summaries with their tags, filtered by `q` (words, `tag:`, `status:`) and `tag`.

    Raises:
        OSError: the runs folder cannot be listed.
    """
    folder.refresh()
    summaries = folder.summaries()
    by_run = tags.all()
    wanted = parse_query(_first(query, "q"), _first(query, "tag"))
    shown = None if wanted.empty else _matching(folder, summaries, by_run, wanted, cache)
    runs = [
        {**summary, "tags": by_run.get(summary["key"], [])}
        for summary in summaries
        if shown is None or summary["key"] in shown
    ]
    keys = {summary["key"] for summary in summaries}
    known = sorted({tag for key, found in by_run.items() if key in keys for tag in found})
    return {"runs": runs, "tags": known, "total": len(summaries)}


def _matching(
    folder: RunFolder,
    summaries: list[dict[str, Any]],
    by_run: dict[str, list[str]],
    wanted: Query,
    cache: SearchCache | None,
) -> set[str]:
    def search() -> set[str]:
        folds = folder.search_folds() if wanted.words else {}
        return {
            s["key"]
            for s in summaries
            if matches(s, folds.get(s["key"], ()), by_run.get(s["key"], []), wanted)
        }

    if cache is None:
        return search()
    runs = tuple((s["key"], s["events"], s["status"]) for s in summaries)
    tagged = tuple(sorted((key, tuple(found)) for key, found in by_run.items()))
    return cache.keys((wanted, runs, tagged), search)


class TagsExchange(Protocol):
    """What the tags route needs from the HTTP handler."""

    def refusal(self, content_type: str) -> str: ...

    def raw_body(self) -> bytes | None: ...

    def reply_json(self, status: int, payload: Any) -> None: ...


def tags_request(ex: TagsExchange, folder: RunFolder, tags: TagStore, key: str) -> None:
    """`POST /api/runs/KEY/tags`: Origin, CSRF and JSON checked first, like every POST."""
    refusal = ex.refusal("application/json")
    if refusal:
        logger.warning("viewer: refused a tags request: %s", refusal)
        ex.reply_json(HTTPStatus.FORBIDDEN, {"error": refusal})
        return
    try:
        ex.reply_json(*set_tags(folder, tags, key, ex.raw_body()))
    except OSError as exc:
        log_failure(logger, logging.ERROR, exc, "viewer: tagging run %s failed", key)
        ex.reply_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "the tags were not saved"})


def set_tags(folder: RunFolder, tags: TagStore, key: str, body: bytes | None) -> Reply:
    """Add and remove tags of one run from a JSON body; the run's tags back.

    Raises:
        OSError: the runs folder cannot be listed or the sidecar cannot be written.
    """
    change = _change(body)
    if change is None:
        return HTTPStatus.BAD_REQUEST, {"error": _BODY_HINT}
    folder.refresh()
    if not any(summary["key"] == key for summary in folder.summaries()):
        return HTTPStatus.NOT_FOUND, {"error": "no such run"}
    try:
        found = tags.update(key, add=change[0], remove=change[1])
    except ToolUsageError as exc:
        return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
    return HTTPStatus.OK, {"key": key, "tags": found}


def _change(body: bytes | None) -> tuple[list[Any], list[Any]] | None:
    if not body:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    if not isinstance(data, dict) or not set(data) <= {"add", "remove"} or not data:
        return None
    add, remove = data.get("add", []), data.get("remove", [])
    if not isinstance(add, list) or not isinstance(remove, list):
        return None
    if len(add) > MAX_TAGS or len(remove) > MAX_TAGS:
        return None
    return add, remove


def _first(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name) or [""]
    return values[0]
