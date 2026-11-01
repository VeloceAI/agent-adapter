"""What goes into a turn and how a stored turn reads back: checks, history, state, events."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from agent_adapter.core.errors import log_failure
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ErrorInfo
from agent_adapter.hooks.builtin.logging import REDACTED, is_secret_field
from agent_adapter.viewer.chat.live import LiveTurn
from agent_adapter.viewer.chat.records import Refused, Session, Turn

logger = logging.getLogger(__name__)

MAX_TEXT = 32 * 1024  # UTF-8 bytes of one message
MAX_ATTACHMENTS = 20
_SECRET_PROBE = "x" * 64  # a secret-like value: would the view mask a value under this key?


def history(session: Session, turn: Turn) -> tuple[Message, ...]:
    """Earlier finished turns of the session as user and assistant messages."""
    messages: list[Message] = []
    for earlier in session.turns:
        if earlier is turn:
            break
        if earlier.status == "done":
            messages.append(Message(role="user", content=earlier.text))
            messages.append(Message(role="assistant", content=earlier.answer))
    return tuple(messages)


def replayed(turn: Turn) -> LiveTurn:
    """A stored turn's outcome as an ended event list: its answer, then the end event."""
    live = LiveTurn(turn.id, turn.session_id, turn.run_id)
    if turn.answer:
        live.push("text", text=turn.answer)
    if turn.status == "done":
        live.push("done", status="done", turn=turn.view())
    else:
        live.push("error", status=turn.status, **(turn.error or {}), turn=turn.view())
    return live


def unmask(new: Any, old: Any) -> Any:
    """`new` with each `***` the state view showed replaced by the value it hid in `old`.

    Raises:
        Refused: a `***` under a secret key matches no stored value (400): writing it would
            replace a secret with the mask.
    """
    restored = _restore(new, old)
    if _mask_left(restored):
        raise Refused(
            HTTPStatus.BAD_REQUEST,
            "a masked value (***) matches no stored value: type the secret again",
        )
    return restored


def _restore(new: Any, old: Any) -> Any:
    if isinstance(new, dict):
        stored = old if isinstance(old, dict) else {}
        return {
            key: stored[key]
            if value == REDACTED and key in stored and is_secret_field(key, stored[key])
            else _restore(value, stored.get(key))
            for key, value in new.items()
        }
    if isinstance(new, list):
        olds = old if isinstance(old, list) and len(old) == len(new) else [None] * len(new)
        return [_restore(item, before) for item, before in zip(new, olds, strict=True)]
    return new


def _mask_left(value: Any) -> bool:
    """True when a `***` sits where the view masks secrets."""
    if isinstance(value, dict):
        return any(
            (item == REDACTED and is_secret_field(key, _SECRET_PROBE)) or _mask_left(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_mask_left(item) for item in value)
    return False


def checked_text(text: Any) -> str:
    if not isinstance(text, str):
        raise Refused(HTTPStatus.BAD_REQUEST, "text must be a string")
    if len(text.encode("utf-8", "surrogatepass")) > MAX_TEXT:
        raise Refused(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"a message may hold {MAX_TEXT} bytes")
    return text


def checked_ids(file_ids: Any) -> list[str]:
    ids = [] if file_ids is None else file_ids
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise Refused(HTTPStatus.BAD_REQUEST, "attachment_ids must be a list of file ids")
    if len(ids) > MAX_ATTACHMENTS:
        raise Refused(HTTPStatus.BAD_REQUEST, f"at most {MAX_ATTACHMENTS} attachments")
    return list(dict.fromkeys(ids))


def failure(exc: BaseException, turn: Turn) -> dict[str, str]:
    """A failed turn's error: an AdapterError's kind and text, else only the type name."""
    info = ErrorInfo.from_exception(exc)
    log_failure(logger, logging.WARNING, exc, "chat: turn %s failed (%s)", turn.id, info.kind)
    return {"kind": info.kind, "message": info.message}
