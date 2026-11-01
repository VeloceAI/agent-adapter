"""Shared OpenAPI test document and an httpx MockTransport recorder."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import httpx

from agent_adapter.core.registry import Registry
from agent_adapter.integrations.openapi import import_openapi

PET = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "id": {"type": "integer", "readOnly": True},
        "name": {"type": "string", "description": "Pet name."},
        "age": {"type": "integer"},
        "nickname": {"type": "string", "nullable": True},
    },
}

PETS: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "Pets", "version": "1"},
    "servers": [
        {"url": "https://{host}/v1", "variables": {"host": {"default": "api.example.com"}}}
    ],
    "components": {
        "schemas": {"Pet": PET, "Tags": {"type": "array", "items": {"type": "string"}}},
        "parameters": {
            "PetId": {
                "name": "petId",
                "in": "path",
                "required": True,
                "schema": {"type": "integer"},
            }
        },
        "requestBodies": {
            "NewPet": {
                "required": True,
                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}},
            }
        },
    },
    "paths": {
        "/pets/{petId}": {
            "parameters": [{"$ref": "#/components/parameters/PetId"}],
            "get": {
                "operationId": "getPet",
                "summary": "Get a pet",
                "description": "Fetch one pet by id.",
                "tags": ["pets"],
                "parameters": [
                    {"name": "verbose", "in": "query", "schema": {"type": "boolean"}},
                    {"name": "include", "in": "query", "schema": {"type": "array"}},
                    {
                        "name": "fields",
                        "in": "query",
                        "explode": False,
                        "schema": {"type": "array"},
                    },
                    {"name": "X-Trace", "in": "header", "description": "Trace id."},
                    {"name": "session", "in": "cookie", "schema": {"type": "string"}},
                    {"name": "Accept", "in": "header", "schema": {"type": "string"}},
                ],
                "responses": {
                    "200": {
                        "description": "ok",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}
                        },
                    }
                },
            },
            "delete": {"operationId": "deletePet", "responses": {"204": {"description": "gone"}}},
        },
        "/pets": {
            "post": {
                "operationId": "createPet",
                "summary": "Create a pet",
                "parameters": [{"name": "name", "in": "query", "schema": {"type": "string"}}],
                "requestBody": {"$ref": "#/components/requestBodies/NewPet"},
                "responses": {"201": {"description": "created"}},
            },
            "get": {"summary": "List pets", "responses": {"200": {"description": "text"}}},
        },
        "/pets/{petId}/tags": {
            "put": {
                "operationId": "setTags",
                "parameters": [{"$ref": "#/components/parameters/PetId"}],
                "requestBody": {
                    "content": {
                        "application/vnd.pets+json": {
                            "schema": {"$ref": "#/components/schemas/Tags"}
                        }
                    }
                },
                "responses": {"200": {"description": "ok"}},
            }
        },
        "/pets/{petId}/photo": {
            "post": {
                "operationId": "uploadPhoto",
                "parameters": [{"$ref": "#/components/parameters/PetId"}],
                "requestBody": {"content": {"multipart/form-data": {"schema": {"type": "object"}}}},
                "responses": {"200": {"description": "ok"}},
            }
        },
    },
}

Reply = httpx.Response | Callable[[httpx.Request], httpx.Response]


def raw(status: int, data: bytes, **headers: str) -> httpx.Response:
    """A response whose body is sent exactly as given (httpx decodes `content=` eagerly)."""
    headers = {key.replace("_", "-"): value for key, value in headers.items()}
    return httpx.Response(status, headers=headers, stream=httpx.ByteStream(data))


def fresh(reply: httpx.Response) -> httpx.Response:
    """A new response with the same raw body, so a repeated reply can be streamed again."""
    if not isinstance(reply.stream, httpx.ByteStream):
        return reply  # a one-off stream (a test's slow body)
    body = b"".join(reply.stream)
    return httpx.Response(reply.status_code, headers=reply.headers, stream=httpx.ByteStream(body))


class Wire:
    """Answers each request with the next reply (the last one repeats); records requests."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies) or [httpx.Response(200, json={"ok": True})]
        self.requests: list[httpx.Request] = []
        self.clients: list[dict[str, Any]] = []  # client_factory keyword arguments
        self.closed = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply(request) if callable(reply) else fresh(reply)

    def factory(self, **options: Any) -> httpx.AsyncClient:
        self.clients.append(options)
        wire = self

        class Client(httpx.AsyncClient):
            async def aclose(self) -> None:
                wire.closed += 1
                await super().aclose()

        return Client(transport=httpx.MockTransport(self), **options)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def body(self, index: int = -1) -> Any:
        return json.loads(self.requests[index].content)


def pets(**options: Any) -> tuple[Registry, Wire]:
    """A registry with the PETS document imported over a fresh `Wire`."""
    wire = options.pop("wire", None) or Wire()
    document = options.pop("document", None) or copy.deepcopy(PETS)
    registry = Registry()
    import_openapi(registry, document, client_factory=wire.factory, **options)
    return registry, wire
