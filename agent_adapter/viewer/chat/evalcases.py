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


class EvalCases:
    """Appends one JSON object per line; case ids stay unique, as `load_cases` requires."""

    def __init__(self, folder: Path) -> None:
        self.path = folder / FILE_NAME
        self._lock = threading.Lock()
        self._ids = self._read_ids()

    def _read_ids(self) -> set[str]:
        if not self.path.is_file():
            return set()
        ids: set[str] = set()
        try:
            text = self.path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            log_failure(logger, logging.WARNING, exc, "chat: cannot read %s", FILE_NAME)
            return ids
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
            Refused: a bad id or tags (400), a duplicate id (409), or a write failure (500).
        """
        if not isinstance(case_id, str) or not case_id.strip() or len(case_id) > MAX_ID:
            raise Refused(HTTPStatus.BAD_REQUEST, f"id must be text of 1-{MAX_ID} characters")
        tags = [] if tags is None else tags
        if (
            not isinstance(tags, list)
            or len(tags) > MAX_TAGS
            or not all(isinstance(tag, str) and tag.strip() for tag in tags)
        ):
            raise Refused(
                HTTPStatus.BAD_REQUEST, f"tags must be a list of at most {MAX_TAGS} names"
            )
        case = {
            "id": case_id,
            "input": turn.text,
            "expected": expected,
            "tags": tags,
            "session_id": turn.session_id,
            "turn_id": turn.id,
            "run_id": turn.run_id,
        }
        line = json.dumps(case, ensure_ascii=True) + "\n"
        with self._lock:
            if case_id in self._ids:
                raise Refused(HTTPStatus.CONFLICT, f"a case with id {case_id!r} exists")
            try:
                with self.path.open("a", encoding="utf-8", newline="\n") as out:
                    out.write(line)
            except OSError as exc:
                log_failure(logger, logging.ERROR, exc, "chat: writing %s failed", FILE_NAME)
                raise Refused(HTTPStatus.INTERNAL_SERVER_ERROR, "the case was not saved") from exc
            self._ids.add(case_id)
        logger.info("chat: saved eval case from turn %s", turn.id)
        return case
