"""Uploads: media type sniffed from the bytes, sizes capped, names kept as display text only."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from http import HTTPStatus
from pathlib import Path

from agent_adapter.core.errors import log_failure
from agent_adapter.viewer.chat.contract import Attachment
from agent_adapter.viewer.chat.records import FileRef, Refused, Session
from agent_adapter.viewer.chat.sessions import SessionStore, new_key

logger = logging.getLogger(__name__)

MAX_FILE = 10 * 1024 * 1024
MAX_SESSION_FILES = 50 * 1024 * 1024
MAX_NAME = 120
_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"%PDF-", "application/pdf"),
)
_TEXT_TYPES = {".md": "text/markdown", ".csv": "text/csv"}
_UNSAFE = re.compile(r"[\x00-\x1f\x7f]")


def sniff(data: bytes, name: str) -> str | None:
    """The media type the bytes show, or None when it is not an allowed type."""
    for magic, media_type in _SIGNATURES:
        if data.startswith(magic):
            return media_type
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if b"\x00" in data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        json.loads(text)
    except ValueError:
        return _TEXT_TYPES.get(Path(name).suffix.lower(), "text/plain")
    return "application/json"


def display_name(raw: str) -> str:
    """The client's file name as safe display text: last path part, no control or format chars."""
    name = re.split(r"[\\/]", unicodedata.normalize("NFC", str(raw or "")))[-1]
    name = _UNSAFE.sub("", name)
    # Format characters (U+202E and the like) can make a name read as another one.
    name = "".join(c for c in name if unicodedata.category(c) != "Cf").strip().lstrip(".")
    return name[:MAX_NAME] or "file"


class FileStore:
    """Stores uploads as `<chat-dir>/<sid>/files/<random id>`; never a client path."""

    def __init__(self, store: SessionStore) -> None:
        self.store = store
        self.root = store.folder

    def add(self, session: Session, raw_name: str, data: bytes) -> FileRef:
        """Store `data` for `session` and record it there.

        Raises:
            Refused: empty or too big (413), a type not allowed (415), or the session is full.
            OSError: the file could not be written.
        """
        if not data:
            raise Refused(HTTPStatus.BAD_REQUEST, "the file is empty")
        if len(data) > MAX_FILE:
            raise Refused(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"a file may hold {MAX_FILE} bytes")
        name = display_name(raw_name)
        media_type = sniff(data, name)
        if media_type is None:
            raise Refused(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "allowed: text, JSON, PNG, JPEG, GIF, WebP and PDF files",
            )
        with self.store.lock:
            if session.uploaded_bytes + len(data) > MAX_SESSION_FILES:
                raise Refused(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    f"a session may hold {MAX_SESSION_FILES} bytes of files",
                )
            session.uploaded_bytes += len(data)
        file_id = new_key("fil_")
        relative = Path(session.id, "files", file_id)
        target = self.root / relative
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError:
            with self.store.lock:
                session.uploaded_bytes -= len(data)
            raise
        ref = FileRef(file_id, name, media_type, len(data), relative.as_posix())
        with self.store.lock:
            session.files[file_id] = ref
        self.store.save(session)
        logger.info(
            "chat: stored %s (%s, %d bytes) in %s", file_id, media_type, ref.size, session.id
        )
        return ref

    def path(self, ref: FileRef) -> Path:
        """The stored file's absolute path, which must stay inside the chat folder.

        Raises:
            Refused: the record points outside the chat folder (404).
        """
        path = (self.root / ref.path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise Refused(HTTPStatus.NOT_FOUND, "no such file")
        return path

    def attachments(self, session: Session, file_ids: list[str]) -> tuple[Attachment, ...]:
        """The session's files with these ids, as the agent gets them.

        Raises:
            Refused: an id the session does not hold (404).
        """
        found: list[Attachment] = []
        for file_id in file_ids:
            with self.store.lock:
                ref = session.files.get(file_id)
            if ref is None:
                raise Refused(HTTPStatus.NOT_FOUND, f"no file {file_id} in this session")
            found.append(Attachment(ref.name, ref.media_type, ref.size, self.path(ref), ref.id))
        return tuple(found)

    def read(self, file_id: str) -> tuple[FileRef, bytes]:
        """A stored file's record and bytes.

        Raises:
            Refused: no such file (404).
        """
        ref = self.store.find_file(file_id)
        path = self.path(ref)
        try:
            return ref, path.read_bytes()
        except OSError as exc:
            log_failure(logger, logging.ERROR, exc, "chat: reading %s failed", file_id)
            raise Refused(HTTPStatus.NOT_FOUND, "the file could not be read") from exc
