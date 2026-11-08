"""Tool listings (HTTP catalogue, MCP hints) show the approvals a policy adds."""

from __future__ import annotations

from agent_adapter import Executor
from agent_adapter.integrations.mcp._handlers import listed
from agent_adapter.policy import load_policy
from tests.server.helpers import call, make_registry, make_service

POLICY = {"rules": [{"tools": "add", "confirm": True}]}


async def test_the_http_catalogue_shows_an_approval_the_policy_adds():
    registry = make_registry()
    executor = load_policy(POLICY).apply(Executor(registry))
    service = make_service(registry, executor=executor)
    reply = await call(service, "GET", "/tools")
    add = next(tool for tool in reply.body["tools"] if tool["name"] == "add")
    assert add["requires_approval"] is True and add["approval"] == "conditional"
    one = await call(service, "GET", "/tools/add")
    assert one.body["requires_approval"] is True


async def test_without_a_policy_the_catalogue_shows_the_code_settings():
    reply = await call(make_service(), "GET", "/tools")
    add = next(tool for tool in reply.body["tools"] if tool["name"] == "add")
    assert add["requires_approval"] is False


def test_mcp_hints_use_the_policy_spec_and_leave_others_alone():
    registry = make_registry()
    spec = registry.get("add")
    governed = load_policy(POLICY).apply(Executor(registry))
    assert listed(governed, spec).confirm is not False
    assert listed(Executor(registry), spec) is spec
    assert listed(object(), spec) is spec  # a runner without a policy
