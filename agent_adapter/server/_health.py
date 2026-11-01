"""Readiness for `/readyz`: 200 once the app can serve, 503 while it starts or drains."""

from __future__ import annotations

import logging

from agent_adapter._version import __version__
from agent_adapter.server._state import ServiceResponse, json_response, problem_response
from agent_adapter.server.problems import problem

logger = logging.getLogger(__name__)


class Readiness:
    """The `/readyz` state of one `create_app` app, found at `app.state.readiness`.

    The lifespan marks it started, and draining when shutdown begins. Hosts that drain on
    their own (a SIGTERM handler, a parent app's lifespan) call `drain()` first.
    """

    def __init__(self, *, needs_lifespan: bool) -> None:
        self._needs_lifespan = needs_lifespan  # MCP serves only inside the lifespan
        self._started = False
        self._draining = False

    @property
    def ready(self) -> bool:
        """True when new requests should be routed here."""
        return not self._draining and (self._started or not self._needs_lifespan)

    @property
    def draining(self) -> bool:
        """True once `drain()` was called or the lifespan began shutting down."""
        return self._draining

    def start(self) -> None:
        """The lifespan started: the app is ready (again)."""
        self._started, self._draining = True, False

    def drain(self) -> None:
        """Turn `/readyz` to 503; other routes keep serving in-flight and new requests."""
        if not self._draining:
            logger.info("draining: /readyz now answers 503")
        self._draining = True

    def response(self) -> ServiceResponse:
        """200 `{"status": "ready", "version"}`, else a 503 `service-unavailable` problem."""
        if self.ready:
            return json_response(200, {"status": "ready", "version": __version__})
        detail = "the server is draining" if self._draining else "the server is starting"
        return problem_response(problem("service-unavailable", detail))


__all__ = ["Readiness"]
