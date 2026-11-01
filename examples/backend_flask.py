"""Flask: one Executor per process, run_sync in sync views, the tool server under /agent."""

import atexit
from typing import Any

from backend_common import (
    Backend,
    Reply,
    ask,
    build_backend,
    decide,
    invoke,
    registry,
    tool_server_options,
    user_for,
)
from flask import Flask, Response, abort, jsonify, request

from agent_adapter import run_sync

TIMEOUT_S = 30  # never let one request hold a worker forever
TOOLS_KEY = "agent_adapter_tools"  # app.extensions key of the mounted tool server


def _json(reply: Reply) -> tuple[Response, int]:
    status, body = reply
    return jsonify(body), status


def _user() -> str:
    """Your auth; here a token lookup (401 otherwise)."""
    user = user_for(request.headers.get("Authorization"))
    if user is None:
        abort(401)
    return user


def _body() -> dict[str, Any]:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        abort(400)
    return body


# --8<-- [start:app]
def make_app(backend: Backend | None = None, *, mount_tools: bool = True) -> Flask:
    """The app factory; `backend` defaults to one built here, once per worker process."""
    service = backend if backend is not None else build_backend()
    app = Flask(__name__)
    app.extensions["agent_adapter"] = service

    @app.post("/tools/<name>")
    def call_tool(name: str) -> tuple[Response, int]:
        _user()
        arguments = _body().get("arguments", {})
        return _json(run_sync(invoke(service, name, arguments), timeout=TIMEOUT_S))

    @app.post("/approvals/<approval_id>/decide")
    def decide_call(approval_id: str) -> tuple[Response, int]:
        user, body = _user(), _body()
        work = decide(
            service,
            approval_id,
            reviewer=user,
            approved=body.get("approved") is True,
            edited_args=body.get("edited_arguments"),
        )
        return _json(run_sync(work, timeout=TIMEOUT_S))

    @app.post("/ask")
    def ask_model() -> tuple[Response, int]:
        _user()
        return _json(run_sync(ask(service, str(_body().get("question", ""))), timeout=TIMEOUT_S))

    if mount_tools:
        _mount_tool_server(app, service)
    return app


# --8<-- [end:app]


# --8<-- [start:mount]
def _mount_tool_server(app: Flask, service: Backend) -> None:
    """Serve the REST tool server under /agent (needs agent-adapter[server-wsgi])."""
    from werkzeug.middleware.dispatcher import DispatcherMiddleware

    from agent_adapter.server import create_wsgi_app

    tools = create_wsgi_app(registry, mcp=False, **tool_server_options(service))
    app.extensions[TOOLS_KEY] = tools  # for code that shuts the app down early (tests)
    atexit.register(tools.close)  # end its lifespan and loop thread when the worker exits
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/agent": tools})  # type: ignore[method-assign]


# --8<-- [end:mount]


if __name__ == "__main__":
    make_app().run(port=8000)  # development server; use gunicorn/waitress in production
