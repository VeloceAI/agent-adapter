from typing import Any

import pytest

from agent_adapter._version import __version__
from agent_adapter.core.errors import BudgetExceeded, ConfigError, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.server.auth import Principal
from tests.server.helpers import SECRET, call, invoke, make_service


async def test_healthz_needs_no_auth():
    reply = await call(make_service(), "GET", "/healthz", token=None)
    assert (reply.status, reply.body) == (200, {"status": "ok", "version": __version__})
    assert (await call(make_service(), "POST", "/healthz")).status == 405


async def test_every_other_route_needs_a_principal():
    service = make_service()
    for method, path in [
        ("GET", "/tools"),
        ("GET", "/tools/add"),
        ("POST", "/tools/add/invoke"),
        ("GET", "/approvals"),
        ("GET", "/approvals/apr_x"),
        ("POST", "/approvals/apr_x/decide"),
        ("GET", "/runs/r1/events"),
        ("GET", "/runs/r1/cost"),
        ("GET", "/openapi.json"),
    ]:
        reply = await call(service, method, path, token="wrong")
        assert reply.status == 401, path
        assert reply.media_type == "application/problem+json"
        assert reply.headers["WWW-Authenticate"] == "Bearer"
        assert reply.body["type"] == "urn:agent-adapter:error:unauthorized"


async def test_unknown_routes_and_methods():
    service = make_service()
    assert (await call(service, "GET", "/nope")).status == 404
    assert (await call(service, "DELETE", "/tools")).status == 405
    assert (await call(service, "GET", "/tools/add/invoke")).status == 405


async def test_catalogue_lists_http_tools_with_an_etag():
    service = make_service()
    reply = await call(service, "GET", "/tools")
    assert reply.status == 200 and reply.body["version"] == "1"
    names = [tool["name"] for tool in reply.body["tools"]]
    assert "mcp_only" not in names and names[:2] == ["add", "send_email"]
    add = reply.body["tools"][0]
    assert add["approval"] == "never" and add["requires_approval"] is False
    assert add["effects"] == "read_only" and add["parallel_safe"] is True
    assert "provider" not in add
    send = reply.body["tools"][1]
    assert (send["approval"], send["requires_approval"]) == ("always", True)
    again = await call(service, "GET", "/tools", headers={"If-None-Match": reply.headers["ETag"]})
    assert again.status == 304 and again.body is None


@pytest.mark.parametrize(
    ("shape", "key"), [("openai", "function"), ("anthropic", "input_schema"), ("gemini", "name")]
)
async def test_catalogue_provider_shapes(shape: str, key: str):
    reply = await call(make_service(), "GET", "/tools", query={"shape": shape})
    assert key in reply.body["tools"][0]["provider"]


async def test_catalogue_filters_and_bad_shapes():
    service = make_service()
    by_tag = await call(service, "GET", "/tools", query={"tags": "math"})
    assert [t["name"] for t in by_tag.body["tools"]] == ["add"]
    by_kind = await call(service, "GET", "/tools", query={"kinds": "action,math"})
    assert [t["name"] for t in by_kind.body["tools"]] == ["add", "send_email"]
    assert (await call(service, "GET", "/tools", query={"shape": "xml"})).status == 400


async def test_conditional_approval_and_a_tool_not_exported_to_a_shape():
    reg = Registry()

    @reg.tool(confirm=lambda call, ctx: True, targets=["http", "openai"])
    def maybe() -> str:
        """Sometimes needs approval."""
        return "x"

    service = make_service(reg)
    one = await call(service, "GET", "/tools/maybe")
    assert one.body["approval"] == "conditional" and one.body["requires_approval"] is True
    assert (await call(service, "GET", "/tools/maybe", query={"shape": "openai"})).status == 200
    gemini = await call(service, "GET", "/tools/maybe", query={"shape": "gemini"})
    assert gemini.status == 404
    listed = await call(service, "GET", "/tools", query={"shape": "gemini"})
    assert listed.body["tools"] == []


async def test_get_tool():
    service = make_service()
    reply = await call(service, "GET", "/tools/add", query={"shape": "openai"})
    assert reply.status == 200 and reply.body["provider"]["function"]["name"] == "add"
    assert (await call(service, "GET", "/tools/nope")).status == 404
    assert (await call(service, "GET", "/tools/mcp_only")).status == 404


async def test_authorize_hides_and_forbids_tools():
    async def only_add(principal: Principal, spec: ToolSpec, args: Any) -> bool:
        return spec.name == "add"

    service = make_service(authorize=only_add)
    listed = await call(service, "GET", "/tools")
    assert [t["name"] for t in listed.body["tools"]] == ["add"]
    assert (await call(service, "GET", "/tools/whoami")).status == 403
    denied = await invoke(service, "whoami", {})
    assert denied.status == 403 and denied.body["tool"] == "whoami"
    doc = await call(service, "GET", "/openapi.json")
    assert [p for p in doc.body["paths"] if p.endswith("/invoke")] == ["/tools/add/invoke"]


async def test_authorize_must_return_true_and_a_raising_hook_fails_closed():
    async def truthy(principal: Principal, spec: ToolSpec, args: Any) -> Any:
        return 1

    assert (await invoke(make_service(authorize=truthy), "add", {"a": 1, "b": 2})).status == 403

    async def broken(principal: Principal, spec: ToolSpec, args: Any) -> bool:
        raise RuntimeError(SECRET)

    reply = await invoke(make_service(authorize=broken), "add", {"a": 1, "b": 2})
    assert reply.status == 500 and SECRET not in str(reply.body)


async def test_invoke_ok():
    reply = await invoke(make_service(), "add", {"a": 2, "b": 3})
    assert reply.status == 200
    body = reply.body
    assert (body["status"], body["result"], body["cached"], body["attempts"]) == ("ok", 5, False, 1)
    assert body["run_id"].startswith("run_") and body["call_id"].startswith("call_")


@pytest.mark.parametrize(
    ("tool", "arguments", "status", "kind"),
    [
        ("add", {"a": "x", "b": 1}, 422, "tool-args-invalid"),
        ("nope", {}, 404, "tool-not-found"),
        ("mcp_only", {}, 404, "tool-not-found"),
        ("boom", {"token": SECRET}, 502, "tool-failed"),
        ("refuse_it", {}, 502, "tool-error"),
        ("slow", {}, 504, "tool-timeout"),
    ],
)
async def test_invoke_failures_map_to_problems(
    tool: str, arguments: dict[str, Any], status: int, kind: str
):
    reply = await invoke(make_service(), tool, arguments)
    assert reply.status == status
    assert reply.body["type"] == f"urn:agent-adapter:error:{kind}"
    assert SECRET not in str(reply.body)


async def test_non_json_results(monkeypatch: pytest.MonkeyPatch):
    service = make_service()
    nan = await invoke(service, "nan", {})
    assert (nan.status, nan.body["result"]) == (200, None)  # pydantic JSON mode: NaN -> null

    def unencodable(value: Any) -> Any:
        raise TypeError("no")

    monkeypatch.setattr("agent_adapter.server._calls.to_jsonable", unencodable)
    reply = await invoke(service, "add", {"a": 1, "b": 2})
    assert reply.status == 500 and reply.body["detail"] == "the tool result is not JSON"


async def test_tool_args_invalid_lists_errors_and_tool_error_keeps_its_message():
    service = make_service()
    bad = await invoke(service, "add", {"a": "x", "b": 1})
    assert bad.body["tool"] == "add" and bad.body["errors"][0]["loc"] == ["a"]
    refused = await invoke(service, "refuse_it", {})
    assert refused.body["detail"] == "nope, not today"


async def test_once_per_run_conflicts_in_the_same_run():
    service = make_service()
    first = await invoke(service, "charge", {"amount": 1}, extra={"run_id": "r1"})
    second = await invoke(service, "charge", {"amount": 2}, extra={"run_id": "r1"})
    assert (first.status, second.status) == (200, 409)
    assert second.body["type"] == "urn:agent-adapter:error:run-limit-exceeded"


async def test_budget_exceeded_is_429_with_retry_after():
    reg = Registry()

    @reg.tool
    def spend() -> str:
        """Over budget."""
        raise BudgetExceeded(1.0, 2.0)

    reply = await invoke(make_service(reg), "spend", {})
    assert (reply.status, reply.headers["Retry-After"]) == (429, "60")


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"[1]",
        b'{"arguments": [1]}',
        b'{"arguments": {}, "extra": 1}',
        b"{}",
        b'{"arguments": {"a": NaN}}',
        b"\xff\xfe",
        b'{"arguments": {}, "run_id": 7}',
        b'{"arguments": {}, "run_id": "a/b"}',
        b'{"arguments": {}, "thread_id": "\\n"}',
        b"[" * 100_000 + b"]" * 100_000,
        b'{"arguments": {"a": ' + b"[" * 300 + b"]" * 300 + b"}}",
    ],
)
async def test_malformed_bodies_are_400(raw: bytes):
    reply = await call(make_service(), "POST", "/tools/add/invoke", raw=raw)
    assert reply.status == 400
    assert reply.body["type"] == "urn:agent-adapter:error:bad-request"


async def test_oversized_bodies_are_413_before_auth():
    service = make_service()
    service.max_body_bytes = 64
    reply = await call(service, "POST", "/tools/add/invoke", raw=b" " * 65, token=None)
    assert reply.status == 413


async def test_openapi_route():
    reply = await call(make_service(), "GET", "/openapi.json")
    assert reply.status == 200 and "/tools/add/invoke" in reply.body["paths"]
    assert make_service().openapi(server_url="http://x")["servers"] == [{"url": "http://x"}]


def test_construction_fails_fast():
    with pytest.raises(ConfigError, match="authenticate"):
        make_service(authenticate=None)
    with pytest.raises(ConfigError, match="idempotency_ttl"):
        make_service(idempotency_ttl=0)
    with pytest.raises(ToolUsageError):
        make_service(registry="nope")  # type: ignore[arg-type]
    for bad in ({"executor": object()}, {"approvals": object()}, {"events": object()}):
        with pytest.raises(ToolUsageError):
            make_service(**bad)
