"""Save chat turns as eval cases in `<chat-dir>/eval_cases.jsonl` (read by `load_cases`)."""

from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from pathlib import Path
from typing import Any

from agent_adapter.core.errors import log_failure
from agent_adapter.viewer.chat.records import Refused, Turn

logger = logging.getLogger(__name__)

FILE_NAME = "eval_cases.jsonl"
MAX_ID = 200
MAX_TAGS = 20
MAX_TAG = 100  # characters of one tag
MAX_FILE = 16_000_000  # bytes; a bigger file refuses new cases


class EvalCases:
    """Appends one JSON object per line; case ids stay unique, as `load_cases` requires."""

    def __init__(self, folder: Path) -> None:
        self.path = folder / FILE_NAME
        self._lock = threading.Lock()
        self._seen: tuple[int, int] | None = None  # (size, mtime_ns) the ids were read at
        self._ids: set[str] | None = set()
        self._refresh()

    def _refresh(self) -> None:
        """Read the ids again when another writer changed the file; None when unreadable."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._seen, self._ids = None, set()
            return
        except OSError as exc:
            log_failure(logger, logging.WARNING, exc, "chat: cannot read %s", FILE_NAME)
            self._seen, self._ids = None, None
            return
        stamp = (stat.st_size, stat.st_mtime_ns)
        if stamp != self._seen or self._ids is None:
            self._ids = self._read_ids()
            self._seen = stamp if self._ids is not None else None

    def _read_ids(self) -> set[str] | None:
        ids: set[str] = set()
        try:
            text = self.path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            log_failure(logger, logging.WARNING, exc, "chat: cannot read %s", FILE_NAME)
            return None
        for number, line in enumerate(text.split("\n"), 1):
            if not line.strip():
                continue
            try:
                ids.add(str(json.loads(line)["id"]))
            except (ValueError, KeyError, TypeError) as exc:
                log_failure(
                    logger, logging.WARNING, exc, "chat: %s:%d is not a case", FILE_NAME, number
                )
        return ids

    def add(self, turn: Turn, case_id: Any, expected: Any, tags: Any) -> dict[str, Any]:
        """Append the turn's input as a case; the saved case.

        Raises:
            Refused: a bad id or tags (400), a duplicate id (409), a full file (413), or a
                write failure (500).
        """
        case = {
            "id": check_id(case_id),
            "input": turn.text,
            "expected": expected,
            "tags": check_tags(tags),
            "session_id": turn.session_id,
            "turn_id": turn.id,
            "run_id": turn.run_id,
        }
        self.append(case)
        logger.info("chat: saved eval case from turn %s", turn.id)
        return case

    def append(self, case: dict[str, Any]) -> None:
        """Append one case object whose `id` is new.

        Raises:
            Refused: a duplicate id (409), a full file (413), or a write failure (500).
        """
        case_id = str(case["id"])
        line = json.dumps(case, ensure_ascii=True, default=str) + "\n"
        with self._lock:
            self._refresh()
            if self._ids is None:
                raise Refused(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"{FILE_NAME} cannot be read, so its case ids are unknown: not saved",
                )
            if case_id in self._ids:
                raise Refused(HTTPStatus.CONFLICT, f"a case with id {case_id!r} exists")
            if self._size() + len(line) > MAX_FILE:
                raise Refused(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    f"{FILE_NAME} is full ({MAX_FILE // 1_000_000} MB); start a new file",
                )
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as out:
                    out.write(line)
            except OSError as exc:
                log_failure(logger, logging.ERROR, exc, "chat: writing %s failed", FILE_NAME)
                raise Refused(HTTPStatus.INTERNAL_SERVER_ERROR, "the case was not saved") from exc
            self._ids.add(case_id)
            self._seen = self._stamp()

    def _stamp(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except OSError:
            return None  # read again on the next write
        return stat.st_size, stat.st_mtime_ns

    def _size(self) -> int:
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0


def check_id(case_id: Any) -> str:
    """A case id: text of 1 to `MAX_ID` characters.

    Raises:
        Refused: anything else (400).
    """
    if not isinstance(case_id, str) or not case_id.strip() or len(case_id) > MAX_ID:
        raise Refused(HTTPStatus.BAD_REQUEST, f"id must be text of 1-{MAX_ID} characters")
    return case_id


def check_tags(tags: Any) -> list[str]:
    """Case tags: a list of at most `MAX_TAGS` non-empty names (none for None).

    Raises:
        Refused: anything else (400).
    """
    tags = [] if tags is None else tags
    if (
        not isinstance(tags, list)
        or len(tags) > MAX_TAGS
        or not all(isinstance(tag, str) and tag.strip() and len(tag) <= MAX_TAG for tag in tags)
    ):
        raise Refused(
            HTTPStatus.BAD_REQUEST,
            f"tags must be a list of at most {MAX_TAGS} names of 1-{MAX_TAG} characters",
        )
    return tags
