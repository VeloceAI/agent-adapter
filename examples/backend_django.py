"""Django in one file: async views await the service, sync views use run_sync, tools beside it."""

import json
from collections.abc import Callable
from typing import Any, TypeVar

import django
from backend_common import (
    Reply,
    ask,
    build_backend,
    decide,
    invoke,
    registry,
    tool_server_options,
    user_for,
)
from django.conf import settings

if not settings.configured:  # a real project keeps this in settings.py
    settings.configure(
        SECRET_KEY="example-only",
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=["localhost", "127.0.0.1"],
        MIDDLEWARE=[],  # token-authenticated JSON API: no sessions, so no CSRF middleware
    )
    django.setup()

from django.http import HttpRequest, JsonResponse
from django.urls import path
from django.views.decorators.http import require_POST

from agent_adapter import run_sync

TIMEOUT_S = 30
ViewT = TypeVar("ViewT", bound=Callable[..., Any])

# --8<-- [start:backend]
BACKEND = build_backend()  # module import: one per worker process
# --8<-- [end:backend]


def _json(reply: Reply) -> JsonResponse:
    status, body = reply
    return JsonResponse(body, status=status)


def _read(request: HttpRequest) -> tuple[str, dict[str, Any]] | JsonResponse:
    """The caller and the JSON body, or the 401 / 400 to send (your auth: a token lookup)."""
    user = user_for(request.headers.get("Authorization"))
    if user is None:
        return JsonResponse({"detail": "unknown token"}, status=401)
    try:
        body = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"detail": "the body must be JSON"}, status=400)
    if not isinstance(body, dict):
        return JsonResponse({"detail": "the body must be a JSON object"}, status=400)
    return user, body


def post_only(view: ViewT) -> ViewT:
    """`require_POST` that keeps the view's type for mypy (Django's decorators are untyped)."""
    checked: ViewT = require_POST(view)
    return checked


# --8<-- [start:views]
@post_only
async def call_tool(request: HttpRequest, name: str) -> JsonResponse:
    """An async view: await the service on Django's event loop."""
    read = _read(request)
    if isinstance(read, JsonResponse):
        return read
    _, body = read
    return _json(await invoke(BACKEND, name, body.get("arguments", {})))


@post_only
def decide_call(request: HttpRequest, approval_id: str) -> JsonResponse:
    """A sync view (WSGI): run_sync runs the coroutine on a fresh loop."""
    read = _read(request)
    if isinstance(read, JsonResponse):
        return read
    user, body = read
    work = decide(
        BACKEND,
        approval_id,
        reviewer=user,
        approved=body.get("approved") is True,
        edited_args=body.get("edited_arguments"),
    )
    return _json(run_sync(work, timeout=TIMEOUT_S))


@post_only
async def ask_model(request: HttpRequest) -> JsonResponse:
    read = _read(request)
    if isinstance(read, JsonResponse):
        return read
    _, body = read
    return _json(await ask(BACKEND, str(body.get("question", ""))))


urlpatterns = [
    path("tools/<str:name>", call_tool),
    path("approvals/<str:approval_id>/decide", decide_call),
    path("ask", ask_model),
]
# --8<-- [end:views]


# --8<-- [start:asgi]
def asgi_application() -> Any:
    """Django and the REST tool server in one ASGI app (asgi.py; needs agent-adapter[server])."""
    from django.core.asgi import get_asgi_application
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from agent_adapter.server import create_app

    tools = create_app(registry, mcp=False, **tool_server_options(BACKEND))
    django_app: Any = get_asgi_application()  # an ASGI app; typed loosely by django-stubs
    return Starlette(routes=[Mount("/agent", app=tools), Mount("/", app=django_app)])


# --8<-- [end:asgi]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(asgi_application(), host="127.0.0.1", port=8000)
