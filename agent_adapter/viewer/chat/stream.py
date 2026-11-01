"""Server-Sent Events framing for one turn's event list (shapes: see `routes`)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import log_failure
from agent_adapter.viewer.chat.live import LiveTurn

if TYPE_CHECKING:
    from agent_adapter.viewer.chat.routes import Exchange

logger = logging.getLogger(__name__)

PING = b": ping\n\n"
CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)


def frame(event: dict[str, Any]) -> bytes:
    """One SSE frame: `id`, `event` and a one-line ASCII JSON `data` field.

    ASCII escapes keep U+2028 and lone surrogates from breaking a reader or the encoder.
    """
    data = json.dumps(event, ensure_ascii=True, default=str)
    return f"id: {event['seq']}\nevent: {event['type']}\ndata: {data}\n\n".encode("ascii")


def stream_turn(ex: Exchange, live: LiveTurn, *, since: int, heartbeat_s: float) -> None:
    """Write the turn's events from `since` until its end event, a ping when idle.

    A `since` past the end of an ended turn gets the end event again, so every stream ends
    with `done` or `error`.
    """
    ex.begin_stream()
    sent = since
    try:
        while True:
            events, ended = live.read(sent, heartbeat_s)
            if not events and ended and live.events:
                events = live.events[-1:]
            if events:
                ex.wfile.write(b"".join(frame(event) for event in events))
                sent += len(events)
            elif not ended:
                ex.wfile.write(PING)
            ex.wfile.flush()
            if ended:  # close() ends every turn, so this is reached then too
                return
    except CLIENT_GONE as exc:
        # The page closed the stream (another turn, a reload): it reconnects with `since`.
        logger.debug(
            "chat: stream of %s closed by the client (%s)", live.turn_id, type(exc).__name__
        )
    except Exception as exc:  # the head is sent: no error reply fits, so end the stream
        log_failure(logger, logging.ERROR, exc, "chat: stream of %s failed", live.turn_id)
