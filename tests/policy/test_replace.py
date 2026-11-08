"""A policy is an executor setting: no `replace(...)` drops its gate, approvals, hooks or limits."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ApprovalDenied,
    GuardrailBlocked,
    RunLimitExceeded,
    ToolUsageError,
)
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.exec import Executor, RunLimits, run_tool_loop
from agent_adapter.policy import PolicyDenied, load_policy
from agent_adapter.server.auth import bearer_auth
from tests.fakes.models import ScriptedModel, calls, text
from tests.policy.helpers import Approvals, shop
from tests.server.helpers import TOKENS, invoke, make_service

POLICY = {
    "rules": [
        {"tools": "wipe_orders", "allow": False},
        {"tools": "refund", "confirm": "amount > 100"},
    ],
    "guards": [{"guard": "pii", "on": ["args", "llm_request"]}],
    "limits": {"max_calls": 2},
}
SMALL = {"order_id": "o-1", "amount": 5.0}
BIG = {"order_id": "o-1", "amount": 500.0}


def governed(**options: Any) -> Executor:
    return load_policy(POLICY).apply(Executor(shop(), **options))


def stripped(**options: Any) -> Executor:
    """The governed executor after every replace a caller might do to it."""
    return governed(**options).replace(hooks=[], limits=RunLimits(), namespace="other")


async def check_enforced(executor: Executor) -> None:
    with pytest.raises(PolicyDenied):
        await executor.run("wipe_orders")
    with pytest.raises(ApprovalDenied, match="no approval port"):
        await executor.run("refund", BIG)
    with pytest.raises(GuardrailBlocked):
        await executor.run("send_note", {"text": "mail ann@example.com"})


@pytest.mark.parametrize(
    "changes",
    [{"hooks": []}, {"namespace": "other"}, {"limits": RunLimits()}, {"approval": None}],
)
async def test_replace_never_drops_the_policy(changes: dict[str, Any]) -> None:
    executor = governed().replace(**changes)
    assert executor.policy is not None
    await check_enforced(executor)


async def test_replaced_hooks_hold_only_the_callers_own() -> None:
    executor = governed()
    assert executor.hooks == ()
    again = executor.replace(hooks=(*executor.hooks,))
    assert again.hooks == ()
    await check_enforced(again)


async def test_limits_survive_replace() -> None:
    executor = stripped()
    assert executor.limits.max_calls == 2
    ctx = ToolContext(run_id="r1")
    await executor.run("lookup_order", {"order_id": "a"}, ctx)
    await executor.run("lookup_order", {"order_id": "b"}, ctx)
    with pytest.raises(RunLimitExceeded):
        await executor.run("lookup_order", {"order_id": "c"}, ctx)


async def test_model_calls_keep_policy_guards_after_replace() -> None:
    request = LLMRequest(messages=(Message(role="user", content="mail ann@example.com"),))
    llm = stripped().model(ScriptedModel([text("x")])).bind(ToolContext(run_id="r"))
    with pytest.raises(GuardrailBlocked):
        await llm.generate(request)


async def test_only_policy_none_removes_it() -> None:
    plain = governed().replace(policy=None)
    assert plain.policy is None
    assert await plain.run("wipe_orders") == "wiped"


async def test_tool_loop_with_a_replaced_executor() -> None:
    log: list[str] = []
    executor = load_policy(POLICY).apply(Executor(shop(log))).replace(hooks=[], namespace="x")
    model = ScriptedModel([calls(("wipe_orders", {}), ("refund", BIG)), text("done")])
    result = await run_tool_loop(
        model, executor.registry, [Message(role="user", content="go")], executor=executor
    )
    kinds = [(r.name, r.error.kind if r.error else "") for r in result.tool_results]
    assert kinds == [("wipe_orders", "approval-denied"), ("refund", "approval-denied")]
    assert log == []


async def test_http_service_with_a_replaced_executor() -> None:
    service = make_service(shop(), executor=stripped())
    assert (await invoke(service, "wipe_orders", {})).status == 403
    assert (await invoke(service, "refund", SMALL)).status == 200
    assert (await invoke(service, "refund", BIG)).status == 202


async def test_create_app_with_a_replaced_executor() -> None:
    pytest.importorskip("starlette")
    httpx2 = pytest.importorskip("httpx2")
    from agent_adapter.server import create_app

    app = create_app(shop(), executor=stripped(), authenticate=bearer_auth(TOKENS), mcp=False)
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
            headers={"Authorization": "Bearer t-alice"},
        ) as http,
    ):
        denied = await http.post("/tools/wipe_orders/invoke", json={"arguments": {}})
        parked = await http.post("/tools/refund/invoke", json={"arguments": BIG})
    assert denied.status_code == 403
    assert parked.status_code == 202


async def test_mcp_server_with_a_replaced_executor() -> None:
    pytest.importorskip("mcp")
    from mcp import Client

    from agent_adapter.integrations.mcp import ERROR_KIND_META, mcp_server

    executor = stripped(approval=DeferredApproval(MemoryApprovalStore()))
    async with Client(mcp_server(shop(), executor=executor)) as client:
        denied = await client.call_tool("wipe_orders", {})
        big = await client.call_tool("refund", BIG)
    assert denied.meta is not None and denied.meta[ERROR_KIND_META] == "approval-denied"
    assert big.meta is not None and big.meta[ERROR_KIND_META] == "approval-pending"


async def test_approvals_still_reach_the_port_after_replace() -> None:
    port = Approvals()
    executor = governed().replace(hooks=[], approval=port)
    assert await executor.run("refund", BIG) == "refunded"
    assert [r.tool for r in port.asked] == ["refund"]


def test_a_policy_with_bad_limits_is_refused() -> None:
    class BadLimits:
        def hooks(self) -> tuple[Any, ...]:
            return ()

        def limits(self, current: RunLimits) -> Any:
            return 3

        def effective(self, spec: Any) -> Any:
            return spec

        def describe(self, spec: Any) -> list[str]:
            return []

    with pytest.raises(ToolUsageError, match="RunLimits"):
        Executor(shop(), policy=BadLimits())
