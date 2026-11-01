"""HTTP tool server: the registry over REST + SSE + OpenAPI, with MCP mounted beside it."""

from agent_adapter.server._health import Readiness
from agent_adapter.server.asgi import create_app, serve
from agent_adapter.server.auth import (
    APPROVER_SCOPE,
    Authenticator,
    Authorizer,
    DecisionAuthorizer,
    Principal,
    RequestInfo,
    bearer_auth,
    is_invoker,
    set_log_key,
)
from agent_adapter.server.events import EventStream, MemoryEventStream
from agent_adapter.server.idempotency import (
    Claim,
    IdempotencyStore,
    MemoryIdempotencyStore,
    RedisIdempotencyStore,
    Stored,
)
from agent_adapter.server.openapi import openapi_document
from agent_adapter.server.problems import PROBLEM_MEDIA_TYPE, Problem, problem_for
from agent_adapter.server.service import ServiceResponse, ToolService
from agent_adapter.server.wsgi import create_wsgi_app

__all__ = [
    "APPROVER_SCOPE",
    "PROBLEM_MEDIA_TYPE",
    "Authenticator",
    "Authorizer",
    "Claim",
    "DecisionAuthorizer",
    "EventStream",
    "IdempotencyStore",
    "MemoryEventStream",
    "MemoryIdempotencyStore",
    "Principal",
    "Problem",
    "Readiness",
    "RedisIdempotencyStore",
    "RequestInfo",
    "ServiceResponse",
    "Stored",
    "ToolService",
    "bearer_auth",
    "create_app",
    "create_wsgi_app",
    "is_invoker",
    "openapi_document",
    "problem_for",
    "serve",
    "set_log_key",
]
