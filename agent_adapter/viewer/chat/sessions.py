"""The session store: sessions in memory, each saved as `<chat-dir>/<sid>.json` (atomic)."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import secrets
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.core.types import to_jsonable
from agent_adapter.viewer.chat.records import FileRef, Refused, Session, Turn, session_from_dict

logger = logging.getLogger(__name__)

MAX_SESSIONS = 1_000
MAX_TURNS = 500  # per session
MAX_TITLE = 200
SESSION_FILE = re.compile(r"^ses_[0-9a-f]{16}\.json$")


def new_key(prefix: str) -> str:
    """`prefix` + 16 random hex characters: safe in a URL and as a file name."""
    return prefix + secrets.token_hex(8)


class SessionStore:
    """Thread-safe sessions of one user; every change is saved by `save` from its thread.

    Session files of other users stay on disk, unseen: never loaded, listed or changed.

    Raises:
        ConfigError: the chat folder cannot be made or listed.
    """

    def __init__(self, folder: str | os.PathLike[str], *, user_id: str = "") -> None:
        self.folder = Path(folder).resolve()
        self.user_id = user_id
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"cannot create the chat folder {self.folder}: {exc}") from exc
        self.lock = threading.RLock()
        self._write = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._load()

    def _load(self) -> None:
        try:
            names = sorted(p for p in self.folder.iterdir() if SESSION_FILE.match(p.name))
        except OSError as exc:
            raise ConfigError(f"cannot list the chat folder {self.folder}: {exc}") from exc
        skipped = 0
        for path in names:
            try:
                session, interrupted = session_from_dict(json.loads(path.read_text("utf-8")))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                log_failure(
                    logger, logging.WARNING, exc, "chat: skipped session file %s", path.name
                )
                continue
            if f"{session.id}.json" != path.name:
                logger.warning("chat: skipped %s: it holds session %s", path.name, session.id)
                continue
            if session.user_id != self.user_id:
                skipped += 1
                continue
            self._sessions[session.id] = session
            if interrupted:
                self.save(session)
        if skipped:
            logger.info("chat: %d session(s) of other users left unseen", skipped)
        logger.info("chat: %d session(s) loaded from %s", len(self._sessions), self.folder)

    def create(
        self,
        title: str = "",
        state: dict[str, Any] | None = None,
        *,
        parent: dict[str, str] | None = None,
    ) -> Session:
        """A new, saved session.

        Raises:
            Refused: too many sessions.
        """
        with self.lock:
            if len(self._sessions) >= MAX_SESSIONS:
                raise Refused(HTTPStatus.CONFLICT, f"at most {MAX_SESSIONS} sessions")
            saved = to_jsonable(dict(state or {}))
            session = Session(
                id=new_key("ses_"),
                title=str(title or "New chat")[:MAX_TITLE],
                state=copy.deepcopy(saved),
                saved_state=saved,
                parent=parent,
                user_id=self.user_id,
            )
            self._sessions[session.id] = session
        self.save(session)
        return session

    def get(self, session_id: str) -> Session:
        """The session.

        Raises:
            Refused: no such session (404).
        """
        with self.lock:
            found = self._sessions.get(session_id)
        if found is None:
            raise Refused(HTTPStatus.NOT_FOUND, "no such session")
        return found

    def find_turn(self, turn_id: str) -> tuple[Session, Turn]:
        """The session and turn with this turn id.

        Raises:
            Refused: no such turn (404).
        """
        with self.lock:
            for session in self._sessions.values():
                turn = session.turn(turn_id)
                if turn is not None:
                    return session, turn
        raise Refused(HTTPStatus.NOT_FOUND, "no such turn")

    def find_file(self, file_id: str) -> FileRef:
        """A stored file's record, from any session.

        Raises:
            Refused: no such file (404).
        """
        with self.lock:
            for session in self._sessions.values():
                ref = session.files.get(file_id)
                if ref is not None and ref.owned:
                    return ref
        raise Refused(HTTPStatus.NOT_FOUND, "no such file")

    def approval_views(self) -> list[dict[str, Any]]:
        """The stored approvals of every turn, newest turn first (for approval memory)."""
        with self.lock:
            turns = [turn for session in self._sessions.values() for turn in session.turns]
            turns.sort(key=lambda turn: turn.started, reverse=True)
            return [dict(item) for turn in turns for item in turn.approvals]

    def summaries(self) -> list[dict[str, Any]]:
        """Every session's summary, most recently updated first."""
        with self.lock:
            items = [session.summary() for session in self._sessions.values()]
        return sorted(items, key=lambda item: item["updated"], reverse=True)

    def branches(self, session_id: str) -> list[dict[str, Any]]:
        """Sessions forked from this one: id, title and the turn they branched at."""
        with self.lock:
            return [
                {"id": s.id, "title": s.title, "turn_id": s.parent["turn_id"]}
                for s in self._sessions.values()
                if s.parent is not None and s.parent.get("session_id") == session_id
            ]

    def branch(self, session: Session, turn_id: str, title: str) -> Session:
        """A new session holding `session`'s turns before `turn_id`, and that turn's prior state.

        Raises:
            Refused: no such turn in the session (404), or too many sessions.
        """
        with self.lock:
            index = next((n for n, t in enumerate(session.turns) if t.id == turn_id), None)
            if index is None:
                raise Refused(HTTPStatus.NOT_FOUND, "no such turn in this session")
            turns = copy.deepcopy(session.turns[:index])
            state = copy.deepcopy(session.turns[index].state_before)
            files = {
                key: FileRef(**{**vars(ref), "owned": False}) for key, ref in session.files.items()
            }
        branch = self.create(title, state, parent={"session_id": session.id, "turn_id": turn_id})
        with self.lock:
            for turn in turns:
                turn.session_id = branch.id
            branch.turns, branch.files = turns, files
            branch.memory_upto = min(session.memory_upto, index)
        self.save(branch)
        return branch

    def discard(self, session: Session) -> None:
        """Forget a session and remove its file (a branch whose first turn was refused)."""
        with self.lock:
            self._sessions.pop(session.id, None)
        _remove(self.folder / f"{session.id}.json")
        logger.info("chat: discarded session %s", session.id)

    def check_turn_room(self, session: Session) -> None:
        """Raises Refused (409) when the session is full."""
        if len(session.turns) >= MAX_TURNS:
            raise Refused(HTTPStatus.CONFLICT, f"a session holds at most {MAX_TURNS} turns")

    def save(self, session: Session) -> None:
        """Write the session file by atomic replace; a failure is logged, the session stays.

        Never raises: a disk problem must not end a turn or a request.
        """
        with self._write:
            with self.lock:
                session.updated = time.time()
                data = session.to_dict()
            path = self.folder / f"{session.id}.json"
            temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                temp.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
                os.replace(temp, path)
            except (OSError, TypeError, ValueError) as exc:
                log_failure(
                    logger, logging.ERROR, exc, "chat: saving session %s failed", session.id
                )
                _remove(temp)


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log_failure(logger, logging.WARNING, exc, "chat: could not remove %s", path.name)
