# HTTP server

The tool service, its ASGI and WSGI apps, auth and problems. Guide: [HTTP server](../cross-language/http-server.md).

## `agent_adapter.server`

::: agent_adapter.server
    options:
      members: [APPROVER_SCOPE, Authenticator, Authorizer, Claim, DecisionAuthorizer, EventStream, IdempotencyStore, MemoryEventStream, MemoryIdempotencyStore, PROBLEM_MEDIA_TYPE, Principal, Problem, Readiness, RedisIdempotencyStore, RequestInfo, ServiceResponse, Stored, ToolService, bearer_auth, create_app, create_wsgi_app, is_invoker, openapi_document, problem_for, serve, set_log_key]

## `agent_adapter.server.asgi`

::: agent_adapter.server.asgi
    options:
      members: [PRINCIPAL_STATE, mcp_principal, request_info]

Also exported here, documented under the canonical path: [`create_app`](#agent_adapter.server.create_app), [`serve`](#agent_adapter.server.serve).

## `agent_adapter.server.service`

::: agent_adapter.server.service
    options:
      members: [ANONYMOUS, encode_body]

Also exported here, documented under the canonical path: [`ServiceResponse`](#agent_adapter.server.ServiceResponse), [`ToolService`](#agent_adapter.server.ToolService).
