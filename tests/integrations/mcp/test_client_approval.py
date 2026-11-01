"""client_approval=True: the 2026-07-28 input_required elicitation round trip."""

from __future__ import annotations

import json
import secrets
from typing import Any

import pytest

pytest.importorskip("mcp")

import mcp.types as t
from mcp import Client, MCPError

from agent_adapter.approval import MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.mcp import CALL_ID_META, ERROR_KIND_META, RUN_ID_META, mcp_server
from agent_adapter.integrations.mcp._approval import (
    APPROVAL_INPUT_KEY,
    _parse_state,
    decision_from,
)
from agent_adapter.integrations.mcp._scope import scoped_id
from tests.fakes.approval import RecordingApproval, ScriptedApproval

pytestmark = pytest.mark.contract


def make_registry(log: list[str]) -> Registry:
    reg = Registry()

    @reg.tool(confirm=True)
    async def delete(item: str) -> str:
        """Delete an item."""
        log.append(item)
        return f"deleted {item}"

    @reg.tool(confirm=lambda call, ctx: call.arguments["amount"] > 100)
    async def pay(amount: int) -> str:
        """Pay an amount; large ones need approval."""
        log.append(f"pay {amount}")
        return f"paid {amount}"

    @reg.tool(confirm=True)
    async def whoami(ctx: ToolContext) -> dict[str, str]:
        """Report the ids (needs approval)."""
        return {"run_id": ctx.run_id, "call_id": ctx.call_id}

    return reg


class Elicitor:
    """An elicitation callback answering with a fixed result and recording the prompts."""

    def __init__(self, result: t.ElicitResult) -> None:
        self.result = result
        self.prompts: list[t.ElicitRequestFormParams] = []

    async def __call__(self, ctx: Any, params: Any) -> t.ElicitResult:
        self.prompts.append(params)
        return self.result


def accept(approved: bool = True, **extra: Any) -> Elicitor:
    return Elicitor(t.ElicitResult(action="accept", content={"approved": approved, **extra}))


def untouched_port() -> RecordingApproval:
    return RecordingApproval(ScriptedApproval([]))


async def test_accepted_elicitation_runs_the_tool_without_asking_the_port():
    log: list[str] = []
    port, elicit = untouched_port(), accept()
    server = mcp_server(make_registry(log), executor=Executor(approval=port), client_approval=True)
    async with Client(server, elicitation_callback=elicit) as client:
        result = await client.call_tool("delete", {"item": "a"})
    assert not result.is_error and result.structured_content == {"result": "deleted a"}
    assert log == ["a"] and port.requests == []
    (prompt,) = elicit.prompts
    assert "'delete'" in prompt.message and '{"item": "a"}' in prompt.message
    assert prompt.requested_schema["required"] == ["approved"]


@pytest.mark.parametrize(
    ("elicitor", "reason"),
    [
        (accept(False, reason="too risky"), "too risky"),
        (accept(False), "not approved by the MCP client"),
        (Elicitor(t.ElicitResult(action="decline")), "declined by the MCP client"),
        (Elicitor(t.ElicitResult(action="cancel")), "cancelled by the MCP client"),
        (Elicitor(t.ElicitResult(action="accept", content={"approved": "yes"})), "not approved"),
    ],
)
async def test_anything_but_an_explicit_yes_denies(elicitor: Elicitor, reason: str):
    log: list[str] = []
    server = mcp_server(make_registry(log), executor=Executor(), client_approval=True)
    async with Client(server, elicitation_callback=elicitor) as client:
        result = await client.call_tool("delete", {"item": "a"})
    assert result.is_error and log == []
    assert result.meta is not None and result.meta[ERROR_KIND_META] == "approval-denied"
    assert reason in result.content[0].text


async def test_without_elicitation_the_executor_port_decides():
    log: list[str] = []
    port = RecordingApproval(ScriptedApproval([ApprovalDecision(approved=True)]))
    server = mcp_server(make_registry(log), executor=Executor(approval=port), client_approval=True)
    async with Client(server) as client:
        result = await client.call_tool("delete", {"item": "b"})
    assert not result.is_error and log == ["b"]
    assert [r.tool for r in port.requests] == ["delete"]


async def test_legacy_protocol_falls_back_to_the_executor_port():
    log: list[str] = []
    port = RecordingApproval(ScriptedApproval([ApprovalDecision(approved=True)]))
    elicit = accept()
    server = mcp_server(make_registry(log), executor=Executor(approval=port), client_approval=True)
    async with Client(server, mode="legacy", elicitation_callback=elicit) as client:
        result = await client.call_tool("delete", {"item": "c"})
    assert not result.is_error and log == ["c"]
    assert elicit.prompts == [] and len(port.requests) == 1


async def test_client_approval_off_never_elicits():
    log: list[str] = []
    elicit = accept()
    server = mcp_server(make_registry(log), executor=Executor())
    async with Client(server, elicitation_callback=elicit) as client:
        result = await client.call_tool("delete", {"item": "d"})
    assert result.is_error and elicit.prompts == [] and log == []


async def test_confirm_rule_decides_whether_to_ask():
    log: list[str] = []
    elicit = accept()
    server = mcp_server(make_registry(log), executor=Executor(), client_approval=True)
    async with Client(server, elicitation_callback=elicit) as client:
        small = await client.call_tool("pay", {"amount": 5})
        large = await client.call_tool("pay", {"amount": 500})
    assert not small.is_error and not large.is_error
    assert log == ["pay 5", "pay 500"] and len(elicit.prompts) == 1


async def test_invalid_args_fail_before_anyone_is_asked():
    elicit = accept()
    server = mcp_server(make_registry([]), executor=Executor(), client_approval=True)
    async with Client(server, elicitation_callback=elicit) as client:
        result = await client.call_tool("delete", {"item": 3})
    assert result.is_error and elicit.prompts == []
    assert result.meta is not None and result.meta[ERROR_KIND_META] == "tool-args-invalid"


async def test_ids_survive_the_round_trip():
    server = mcp_server(make_registry([]), executor=Executor(), client_approval=True)
    async with Client(server, elicitation_callback=accept()) as client:
        given = await client.call_tool(
            "whoami", {}, meta={RUN_ID_META: "run_given", CALL_ID_META: "call_given"}
        )
        minted = await client.call_tool("whoami", {})
    assert given.structured_content["run_id"] == scoped_id("run_", None, "run_given")
    assert given.structured_content["call_id"] != scoped_id("call_", None, "call_given")
    assert minted.structured_content["run_id"].startswith("run_")
    assert minted.structured_content["call_id"].startswith("call_")


async def _first_round(client: Client, args: dict[str, Any]) -> t.InputRequiredResult:
    first = await client.session.call_tool("delete", args, allow_input_required=True)
    assert isinstance(first, t.InputRequiredResult)
    assert first.input_requests is not None and APPROVAL_INPUT_KEY in first.input_requests
    assert first.request_state is not None and "call_" not in first.request_state  # sealed
    return first


YES = {APPROVAL_INPUT_KEY: t.ElicitResult(action="accept", content={"approved": True})}


async def test_state_is_bound_to_the_arguments_it_approved():
    log: list[str] = []
    server = mcp_server(make_registry(log), executor=Executor(), client_approval=True)
    async with Client(server, elicitation_callback=accept()) as client:
        first = await _first_round(client, {"item": "safe"})
        with pytest.raises(MCPError, match="Invalid or expired requestState"):
            await client.session.call_tool(
                "delete",
                {"item": "everything"},
                input_responses=YES,
                request_state=first.request_state,
            )
        with pytest.raises(MCPError, match="Invalid or expired requestState"):
            await client.session.call_tool(
                "delete", {"item": "safe"}, input_responses=YES, request_state="v1.forged"
            )
        done = await client.session.call_tool(
            "delete", {"item": "safe"}, input_responses=YES, request_state=first.request_state
        )
    assert isinstance(done, t.CallToolResult) and not done.is_error and log == ["safe"]


async def test_a_resume_without_our_response_is_denied():
    log: list[str] = []
    server = mcp_server(make_registry(log), executor=Executor(), client_approval=True)
    async with Client(server, elicitation_callback=accept()) as client:
        first = await _first_round(client, {"item": "x"})
        result = await client.session.call_tool(
            "delete", {"item": "x"}, input_responses={}, request_state=first.request_state
        )
    assert isinstance(result, t.CallToolResult) and result.is_error and log == []
    assert "no approval response" in result.content[0].text


async def test_shared_keys_let_another_process_finish_the_round_trip():
    keys = [secrets.token_hex(32)]
    log: list[str] = []
    shared = MemoryApprovalStore()
    one, two = (
        mcp_server(
            make_registry(log),
            name="tax",
            client_approval=True,
            request_state_keys=keys,
            approval_store=shared,
        )
        for _ in range(2)
    )
    other = mcp_server(make_registry(log), name="tax", client_approval=True)
    async with Client(one, elicitation_callback=accept()) as client:
        first = await _first_round(client, {"item": "k"})
    async with Client(other, elicitation_callback=accept()) as client:
        with pytest.raises(MCPError, match="Invalid or expired requestState"):
            await client.session.call_tool(
                "delete", {"item": "k"}, input_responses=YES, request_state=first.request_state
            )
    async with Client(two, elicitation_callback=accept()) as client:
        done = await client.session.call_tool(
            "delete", {"item": "k"}, input_responses=YES, request_state=first.request_state
        )
    assert isinstance(done, t.CallToolResult) and not done.is_error and log == ["k"]


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "not json",
        "[]",
        json.dumps({"v": 1, "run_id": "r", "call_id": "c", "iat": 1.0}),
        json.dumps({"v": 2, "run_id": "", "call_id": "c", "iat": 1.0}),
        json.dumps({"v": 2, "run_id": "r", "iat": 1.0}),
        json.dumps({"v": 2, "run_id": "r", "call_id": "c"}),
        json.dumps({"v": 2, "run_id": "r", "call_id": "c", "iat": True}),
        '{"v": 2, "run_id": "r", "call_id": "c", "iat": NaN}',
        json.dumps({"v": 2, "run_id": "r", "call_id": "c", "iat": 10**400}),
        json.dumps({"v": 2.0, "run_id": "r", "call_id": "c", "iat": 1.0}),
        json.dumps({"v": "2", "run_id": "r", "call_id": "c", "iat": 1.0}),
    ],
)
def test_parse_state_rejects_anything_but_our_shape(raw: str | None):
    with pytest.raises(MCPError, match="Invalid or expired requestState"):
        _parse_state(raw)


def test_parse_state_accepts_our_shape():
    raw = json.dumps({"v": 2, "run_id": "r", "call_id": "c", "iat": 1.5})
    assert _parse_state(raw) == {"run_id": "r", "call_id": "c", "iat": 1.5}


def test_decision_from_other_responses_denies():
    decision = decision_from(t.ListRootsResult(roots=[]))
    assert not decision.approved and decision.decided_by == "mcp-client"
    assert decision_from(None).reason == "no approval response from the MCP client"
    yes = decision_from(t.ElicitResult(action="accept", content={"approved": True, "reason": 7}))
    assert yes.approved and yes.reason == "" and yes.decided_at > 0


async def test_long_arguments_are_shortened_in_the_prompt():
    elicit = accept()
    server = mcp_server(make_registry([]), executor=Executor(), client_approval=True)
    async with Client(server, elicitation_callback=elicit) as client:
        await client.call_tool("delete", {"item": "x" * 5000})
    (prompt,) = elicit.prompts
    assert prompt.message.endswith(" ...?") and len(prompt.message) < 2100
