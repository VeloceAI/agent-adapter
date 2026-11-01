"""OpenAPI importer contract: requests on the wire, results, auth, status mapping, clients."""

import sys

import pytest

pytest.importorskip("yaml")
httpx = pytest.importorskip("httpx")

from agent_adapter.core.errors import (  # noqa: E402
    ConfigError,
    IntegrationError,
    MissingDependency,
    ToolError,
    ToolFailed,
)
from agent_adapter.core.registry import Registry  # noqa: E402
from agent_adapter.core.tool import build_spec  # noqa: E402
from agent_adapter.exec import Executor  # noqa: E402
from agent_adapter.integrations.discovery import available_integrations, get_integration  # noqa: E402
from agent_adapter.integrations.openapi import (  # noqa: E402
    INTEGRATION,
    BearerAuth,
    HeaderAuth,
    QueryAuth,
    import_openapi,
)
from agent_adapter.runtime.sync import run_sync  # noqa: E402
from tests.integrations.openapi_mb_cases import *  # noqa: E402, F403
from tests.integrations.openapi_round2_cases import *  # noqa: E402, F403
from tests.integrations.openapi_round3_cases import *  # noqa: E402, F403
from tests.integrations.openapi_round4_cases import *  # noqa: E402, F403
from tests.integrations.openapi_round5_cases import *  # noqa: E402, F403
from tests.integrations.openapi_schema_cases import *  # noqa: E402, F403
from tests.integrations.openapi_security_cases import *  # noqa: E402, F403
from tests.integrations.openapi_support import PETS, Wire, pets  # noqa: E402

pytestmark = pytest.mark.contract


async def no_sleep(_: float) -> None:
    return None


def taken(petId: int) -> int:
    """A tool already registered under an imported name."""
    return petId


# --- requests ------------------------------------------------------------------------------


async def test_get_sends_path_query_header_and_cookie_values():
    registry, wire = pets(auth=HeaderAuth("X-API-Key", "k-123"))
    args = {
        "petId": 7,
        "verbose": True,
        "include": ["owner", "vet"],
        "fields": ["name", "age"],
        "X_Trace": "abc",
        "session": "s 1",
    }
    assert await registry.get("getPet").invoke(args) == {"ok": True}
    request = wire.last
    assert request.method == "GET"
    assert request.url.path == "/v1/pets/7"
    assert request.url.host == "api.example.com"
    assert request.url.params.multi_items() == [
        ("verbose", "true"),
        ("include", "owner"),
        ("include", "vet"),
        ("fields", "name,age"),
    ]
    assert request.headers["X-Trace"] == "abc"
    assert request.headers["Cookie"] == "session=s%201"
    assert request.headers["X-API-Key"] == "k-123"


async def test_unset_and_null_parameters_are_not_sent():
    registry, wire = pets()
    await registry.get("getPet").invoke({"petId": 1, "verbose": None})
    assert wire.last.url.query == b""
    assert "X-Trace" not in wire.last.headers and "Cookie" not in wire.last.headers


async def test_path_values_are_percent_encoded():
    document = {
        **PETS,
        "paths": {
            "/files/{name}": {
                "get": {
                    "operationId": "getFile",
                    "parameters": [{"name": "name", "in": "path", "schema": {"type": "string"}}],
                }
            }
        },
    }
    registry, wire = pets(document=document)
    await registry.get("getFile").invoke({"name": "a/b c"})
    assert wire.last.url.raw_path == b"/v1/files/a%2Fb%20c"


async def test_query_objects_are_compact_json_and_explicit_nulls_are_skipped():
    document = {
        **PETS,
        "paths": {
            "/search": {
                "get": {
                    "operationId": "search",
                    "parameters": [
                        {"name": "filter", "in": "query", "schema": {"type": "object"}},
                        {
                            "name": "maybe",
                            "in": "query",
                            "schema": {"type": "string", "nullable": True},
                        },
                        {"name": "$$", "in": "query", "schema": {"type": "integer"}},
                    ],
                }
            }
        },
    }
    registry, wire = pets(document=document)
    spec = registry.get("search")
    assert list(spec.json_schema()["properties"]) == ["filter", "maybe", "p_"]
    await spec.invoke({"filter": {"a": [1, 2]}, "maybe": None, "p_": 3})
    assert wire.last.url.params.multi_items() == [("filter", '{"a":[1,2]}'), ("$$", "3")]


async def test_post_flattens_the_json_body_with_body_prefix_on_a_clash():
    registry, wire = pets()
    spec = registry.get("createPet")
    assert list(spec.json_schema()["properties"]) == ["name", "body_name", "age", "nickname"]
    await spec.invoke({"name": "q", "body_name": "fido", "age": 3, "nickname": None})
    assert wire.last.url.params["name"] == "q"
    assert wire.body() == {"name": "fido", "age": 3, "nickname": None}
    assert wire.last.headers["content-type"] == "application/json"


async def test_a_required_body_is_sent_even_when_no_field_is_set():
    document = {
        **PETS,
        "paths": {
            "/ping": {
                "post": {
                    "operationId": "ping",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"note": {"type": "string"}},
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }
    registry, wire = pets(document=document)
    await registry.get("ping").invoke({})
    assert wire.body() == {}


async def test_an_optional_body_with_nothing_set_is_not_sent():
    document = {
        **PETS,
        "paths": {
            "/ping": {
                "post": {
                    "operationId": "ping",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"note": {"type": "string"}},
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }
    registry, wire = pets(document=document)
    await registry.get("ping").invoke({})
    assert wire.last.content == b"" and "content-type" not in wire.last.headers


async def test_a_non_object_body_is_one_body_field():
    registry, wire = pets()
    await registry.get("setTags").invoke({"petId": 2, "body": ["calm", "old"]})
    assert wire.last.method == "PUT" and wire.body() == ["calm", "old"]


# --- results and status codes -------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (httpx.Response(200, json=[1, 2]), [1, 2]),
        (httpx.Response(200, text="plain words"), "plain words"),
        (httpx.Response(204), None),
        (httpx.Response(200, content=b""), None),
        (
            httpx.Response(200, content=b"{oops", headers={"content-type": "application/json"}),
            "{oops",
        ),
        (
            httpx.Response(
                200, json={"a": 1}, headers={"content-type": "application/problem+json"}
            ),
            {"a": 1},
        ),
    ],
)
async def test_responses_become_json_text_or_none(reply, expected):
    registry, _ = pets(wire=Wire(reply))
    assert await registry.get("deletePet").invoke({"petId": 1}) == expected


async def test_4xx_raises_tool_error_with_status_and_truncated_body_without_retrying():
    wire = Wire(httpx.Response(404, text="missing " + "x" * 600))
    registry, _ = pets(wire=wire, retries=3)
    with pytest.raises(ToolError) as info:
        await Executor(registry, sleep=no_sleep).run("getPet", {"petId": 9})
    message = str(info.value)
    assert message.startswith("HTTP 404 from GET /pets/{petId}: missing x")
    assert len(message) < 600 and info.value.details == {"status": 404}
    assert len(wire.requests) == 1


async def test_3xx_is_a_tool_error_and_redirects_are_not_followed():
    wire = Wire(httpx.Response(302, headers={"location": "https://elsewhere.example"}))
    registry, _ = pets(wire=wire)
    with pytest.raises(ToolError, match="HTTP 302"):
        await registry.get("getPet").invoke({"petId": 9})
    assert len(wire.requests) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_429_and_5xx_are_retried_then_tool_failed(status):
    wire = Wire(httpx.Response(status), httpx.Response(status), httpx.Response(200, json=1))
    registry, _ = pets(wire=wire, retries=1, auth=QueryAuth("api_key", "SECRET"))
    with pytest.raises(ToolFailed) as info:
        await Executor(registry, sleep=no_sleep).run("getPet", {"petId": 9})
    assert info.value.attempts == 2 and len(wire.requests) == 2
    cause = info.value.__cause__
    assert isinstance(cause, httpx.HTTPStatusError) and cause.response.status_code == status
    assert str(cause) == f"HTTP {status} from GET /pets/{{petId}}" and "SECRET" not in str(cause)


async def test_a_retry_that_succeeds_returns_the_value():
    wire = Wire(httpx.Response(503), httpx.Response(200, json={"n": 1}))
    registry, _ = pets(wire=wire, retries=2)
    assert await Executor(registry, sleep=no_sleep).run("getPet", {"petId": 1}) == {"n": 1}


async def test_transport_errors_propagate_and_are_retried():
    def broken(request):
        raise httpx.ConnectError("refused", request=request)

    registry, _ = pets(wire=Wire(broken), retries=1)
    with pytest.raises(ToolFailed) as info:
        await Executor(registry, sleep=no_sleep).run("getPet", {"petId": 1})
    assert isinstance(info.value.__cause__, httpx.ConnectError) and info.value.attempts == 2


# --- auth ----------------------------------------------------------------------------------


async def async_token() -> str:
    return "async-tok"


@pytest.mark.parametrize(
    ("auth", "header"),
    [
        (BearerAuth("tok"), "Bearer tok"),
        (BearerAuth(lambda: "sync-tok"), "Bearer sync-tok"),
        (BearerAuth(async_token), "Bearer async-tok"),
    ],
)
async def test_bearer_auth_sets_the_authorization_header(auth, header):
    registry, wire = pets(auth=auth)
    await registry.get("deletePet").invoke({"petId": 1})
    assert wire.last.headers["Authorization"] == header


async def test_query_auth_adds_a_parameter():
    registry, wire = pets(auth=QueryAuth("api_key", "q-1"))
    await registry.get("deletePet").invoke({"petId": 1})
    assert wire.last.url.params["api_key"] == "q-1"


async def test_a_token_callable_returning_nothing_is_a_config_error():
    registry, wire = pets(auth=BearerAuth(lambda: ""))
    with pytest.raises(ConfigError, match="returned no token"):
        await registry.get("deletePet").invoke({"petId": 1})
    assert wire.requests == []


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: BearerAuth(""), "non-empty token"),
        (lambda: HeaderAuth("", "v"), "non-empty name"),
        (lambda: QueryAuth("k", ""), "non-empty name"),
    ],
)
def test_auth_rejects_empty_values(make, match):
    with pytest.raises(ConfigError, match=match):
        make()


def test_auth_reprs_hide_secrets():
    texts = [
        repr(BearerAuth("s3cr3t")),
        repr(HeaderAuth("X-Key", "s3cr3t")),
        repr(QueryAuth("k", "s3cr3t")),
    ]
    assert texts == [
        "BearerAuth(token=***)",
        "HeaderAuth(name='X-Key', value=***)",
        "QueryAuth(name='k', value=***)",
    ]


# --- clients, options and policy -------------------------------------------------------------


def test_one_client_per_event_loop_closed_when_run_sync_ends():
    registry, wire = pets(headers={"User-Agent": "aa-test"}, timeout=5.0)
    spec = registry.get("deletePet")
    run_sync(spec.invoke({"petId": 1}))
    run_sync(spec.invoke({"petId": 2}))
    assert len(wire.clients) == 2 and wire.closed == 2
    assert wire.clients[0] == {
        "base_url": "https://api.example.com/v1",
        "headers": {"User-Agent": "aa-test"},
        "timeout": 5.0,
    }
    assert wire.last.headers["User-Agent"] == "aa-test"


async def test_the_default_client_is_an_httpx_async_client():
    registry = Registry()
    [spec] = import_openapi(registry, PETS, operations=["deletePet"], base_url="https://h.test/x")
    client = spec.func.clients.get()  # type: ignore[attr-defined]
    try:
        assert isinstance(client, httpx.AsyncClient) and str(client.base_url) == "https://h.test/x/"
    finally:
        await spec.func.clients.aclose()  # type: ignore[attr-defined]


def test_effects_idempotency_kind_and_tags_follow_the_method():
    registry, _ = pets()
    seen = {s.name: (s.effects, s.idempotent, s.kind, sorted(s.tags)) for s in registry}
    assert seen == {
        "getPet": ("read_only", True, "http", ["openapi", "pets"]),
        "deletePet": ("destructive", True, "http", ["openapi"]),
        "createPet": ("additive", False, "http", ["openapi"]),
        "get_pets": ("read_only", True, "http", ["openapi"]),
        "setTags": ("additive", True, "http", ["openapi"]),
    }


def test_policy_options_override_the_defaults():
    registry, _ = pets(operations=["deletePet"], confirm=True, tags={"vet"}, effects="additive")
    spec = registry.get("deletePet")
    assert (spec.confirm, spec.tags, spec.effects) == (True, frozenset({"vet"}), "additive")


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"auth": {"Authorization": "x"}}, "auth must be"),
        ({"timeout": 0}, "timeout must be"),
        ({"timeout": True}, "timeout must be"),
        ({"prefix": 3}, "prefix must be"),
    ],
)
def test_bad_options_are_config_errors(options, match):
    with pytest.raises(ConfigError, match=match):
        import_openapi(Registry(), PETS, **options)


def test_import_is_all_or_nothing():
    registry = Registry()
    registry.add(build_spec(taken, name="deletePet"))
    with pytest.raises(Exception, match="deletePet"):
        import_openapi(registry, PETS, client_factory=Wire().factory)
    assert registry.names() == ["deletePet"]


# --- the integration object -----------------------------------------------------------------


def test_the_integration_is_discoverable_and_import_only():
    available_integrations(refresh=True)
    assert get_integration("openapi") is INTEGRATION
    with pytest.raises(IntegrationError, match="only imports"):
        INTEGRATION.export_tool(build_spec(taken))


def test_import_needs_httpx(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[openapi\]"):
        import_openapi(Registry(), PETS)
