"""`GET /api/chat/memory` and `POST /api/chat/memory/delete`: the chat user's memory records.

GET answers `{"memory": {"enabled", "user"?, "backend"?, "save"?, "every"?}, "records":
[{"id", "text", "kind", "chat", "created_at"}]}`, newest first. POST takes `{"ids": [id,
...]}` and answers `{"deleted": n}`; only records of the server's user can go.
"""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from agent_adapter.core.errors import AdapterError, ToolUsageError
from agent_adapter.viewer.chat.host import ChatHost
from agent_adapter.viewer.chat.records import Refused

logger = logging.getLogger(__name__)

MEMORY_PATH = "/api/chat/memory"
DELETE_PATH = "/api/chat/memory/delete"


def memory_listing(host: ChatHost) -> dict[str, Any]:
    """The user's records, or `enabled: false` when memory is off.

    Raises:
        Refused: the memory backend failed (502).
    """
    if host.memory is None:
        return {"memory": {"enabled": False}, "records": []}
    try:
        records = host.memory.records()
    except AdapterError as exc:
        raise _failed(exc) from exc
    return {"memory": host.memory.info(), "records": records}


def memory_delete(host: ChatHost, body: dict[str, Any]) -> dict[str, Any]:
    """Delete the listed records of the user.

    Raises:
        Refused: memory is off (404), bad ids (400), or the backend failed (502).
    """
    if host.memory is None:
        raise Refused(HTTPStatus.NOT_FOUND, "memory is off")
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids):
        raise Refused(HTTPStatus.BAD_REQUEST, 'send {"ids": ["mem_...", ...]}')
    try:
        return {"deleted": host.memory.delete(ids)}
    except ToolUsageError as exc:
        raise Refused(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    except AdapterError as exc:
        raise _failed(exc) from exc


def _failed(exc: AdapterError) -> Refused:
    logger.error("chat: a memory request failed (%s)", type(exc).__name__)
    return Refused(HTTPStatus.BAD_GATEWAY, "the memory backend failed")
