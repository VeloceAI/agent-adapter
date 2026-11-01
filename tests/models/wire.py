"""A MockTransport handler and SSE bodies shared by the provider port tests (no network)."""

from __future__ import annotations

import json
from typing import Any, ClassVar


class RecordingWire:
    """Records each request's path, URL and JSON body; answers with the next reply.

    A reply is an exception (raised), a ready response, or JSON for a 200. Subclasses set
    `response_type` to the httpx flavour their SDK uses.
    """

    response_type: ClassVar[Any]

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.urls: list[str] = []

    def __call__(self, request: Any) -> Any:
        self.paths.append(request.url.path)
        self.urls.append(str(request.url))
        self.bodies.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, self.response_type):
            return reply
        return self.response_type(200, json=reply)


def typed_sse(response_type: Any, events: list[dict[str, Any]]) -> Any:
    """A 200 event stream with one `event: <type>` / `data: <json>` pair per event."""
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return response_type(200, content=body.encode(), headers={"content-type": "text/event-stream"})
