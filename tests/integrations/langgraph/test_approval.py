"""`LangGraphApproval`: interrupt payloads, resume binding, and fail-closed resume values."""

import pytest

pytest.importorskip("langgraph")

from langgraph.errors import is_invalid_resume
from langgraph.graph import MessagesState
from langgraph.types import Command
from pydantic import ValidationError

from agent_adapter.core.errors import ApprovalPending, ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.integrations.langgraph import LangGraphApproval, one_use
from agent_adapter.integrations.langgraph.approval import PAYLOAD_TYPE, binding, payload
from tests.integrations.langgraph.support import bound, config, one_node, resume

pytestmark = pytest.mark.contract


def request(**changes):
    base = {
        "approval_id": "apr_1",
        "run_id": "t1",
        "call_id": "call_1",
        "tool": "delete",
        "arguments": {"item": "x"},
    }
    return ApprovalRequest(**{**base, **changes})


def test_binding_ignores_the_approval_id_and_time_but_not_the_request():
    first = binding(request())
    assert binding(request(approval_id="apr_2", created_at=5.0, hint="h")) == first
    for changed in (
        request(run_id="t2"),
        request(tool="drop"),
        request(arguments={"item": "y"}),
        request(call_id="call_2"),
        request(kind="node"),
    ):
        assert binding(changed) != first


def test_inexact_binding_ignores_the_call_id_only():
    assert binding(request(), exact=False) == binding(request(call_id="other"), exact=False)
    assert binding(request(), exact=False) != binding(request())
    assert binding(request(), exact=False) != binding(request(run_id="t2"), exact=False)


def test_binding_refuses_arguments_that_are_not_json():
    with pytest.raises(ToolUsageError):
        binding(request(arguments={"raw": b"x"}))


def test_payload_carries_the_request_and_its_binding():
    value = payload(request(), binding(request()))
    assert value["type"] == PAYLOAD_TYPE
    assert value["binding"] == binding(request())
    assert value["request"]["tool"] == "delete" and value["request"]["arguments"] == {"item": "x"}


def _approval_graph(port, decisions):
    async def node(state):
        decisions.append(await port.request(request(run_id="t-graph")))
        return {"messages": []}

    return one_node(one_use(node), MessagesState)


async def test_request_interrupts_and_a_bound_resume_value_decides():
    decisions, port = [], LangGraphApproval()
    app = _approval_graph(port, decisions)
    cfg = config()
    out = await app.ainvoke({"messages": []}, cfg, version="v2")
    [interrupt] = out.interrupts
    assert interrupt.value["request"]["run_id"] == "t-graph"
    assert interrupt.response_schema["required"] == ["binding", "decision", "mac"]
    decision = ApprovalDecision(approved=True, edited_args={"item": "z"}, reason="ok")
    out = await app.ainvoke(resume(interrupt, decision), cfg, version="v2")
    assert not out.interrupts
    assert decisions == [decision]


@pytest.mark.parametrize(
    "value",
    [
        {"approved": True},  # a plain decision binds to nothing
        {"binding": "0" * 64, "decision": {"approved": True}},  # forged
        "yes",
        {"binding": None, "decision": {"approved": True}},
        {"decision": {"approved": True}},
    ],
)
async def test_unbound_or_forged_resume_values_are_refused_and_can_be_answered_again(value):
    decisions, port = [], LangGraphApproval()
    app = _approval_graph(port, decisions)
    cfg = config()
    [interrupt] = (await app.ainvoke({"messages": []}, cfg, version="v2")).interrupts
    with pytest.raises(ValidationError) as caught:
        await app.ainvoke(Command(resume={interrupt.id: value}), cfg, version="v2")
    assert is_invalid_resume(caught.value)
    assert decisions == []
    snapshot = await app.aget_state(cfg)
    assert [i.id for i in snapshot.interrupts] == [interrupt.id]
    await app.ainvoke(resume(interrupt, ApprovalDecision(approved=False)), cfg, version="v2")
    assert decisions == [ApprovalDecision(approved=False)]


async def test_a_value_bound_to_another_request_is_refused():
    decisions, port = [], LangGraphApproval()
    app = _approval_graph(port, decisions)
    cfg = config()
    [interrupt] = (await app.ainvoke({"messages": []}, cfg, version="v2")).interrupts
    other = {"binding": binding(request(run_id="another-thread")), "decision": {"approved": True}}
    with pytest.raises(ValidationError):
        await app.ainvoke(Command(resume={interrupt.id: other}), cfg, version="v2")
    assert decisions == []
    assert bound(interrupt, {"approved": True})["binding"] == binding(request(run_id="t-graph"))


async def test_outside_a_graph_the_port_refuses_to_decide():
    with pytest.raises(ToolUsageError, match="inside a LangGraph node"):
        await LangGraphApproval().request(request())


async def test_inside_a_compiled_flow_node_the_port_defers_to_the_node():
    from agent_adapter.integrations.langgraph.approval import defer_to_node

    async def node(state, config):
        ns = config["metadata"]["langgraph_checkpoint_ns"]
        with defer_to_node(ns), pytest.raises(ApprovalPending) as caught:
            await LangGraphApproval().request(request())
        assert caught.value.tool == "delete" and caught.value.call_id == "call_1"
        assert caught.value.arguments == {"item": "x"}
        return {"messages": []}

    out = await one_node(node).ainvoke({"messages": []}, config(), version="v2")
    assert not out.interrupts


def test_a_port_key_must_be_bytes_of_16_or_more():
    with pytest.raises(ToolUsageError, match="16 bytes"):
        LangGraphApproval(key=b"short")
    assert LangGraphApproval().key == LangGraphApproval().key  # the per-process key
    assert LangGraphApproval(key=b"k" * 16).key == b"k" * 16
    with pytest.raises(ToolUsageError, match="not an agent_adapter approval"):
        LangGraphApproval().resume_value({"type": "other"}, ApprovalDecision(approved=True))


async def test_a_value_langgraph_did_not_validate_is_still_refused(monkeypatch):
    import langgraph.types

    from agent_adapter.integrations.langgraph.approval import ask

    unchecked = {"binding": "x", "decision": {"approved": True}, "mac": "y"}
    monkeypatch.setattr(langgraph.types, "interrupt", lambda value, response_schema=None: unchecked)
    errors: list[Exception] = []

    async def node(state):
        try:
            ask(request(), key=b"k" * 16)
        except ToolUsageError as exc:
            errors.append(exc)
        return {"messages": []}

    await one_node(one_use(node)).ainvoke({"messages": []}, config())
    assert [str(e) for e in errors] == ["the resume value is not bound to this approval request"]


async def test_an_unknown_scratchpad_fails_closed(monkeypatch):
    from agent_adapter.integrations.langgraph import approval

    monkeypatch.setattr(approval, "SCRATCHPAD_KEY", "__renamed_scratchpad__")
    errors: list[Exception] = []

    async def node(state):
        try:
            await LangGraphApproval().request(request())
        except ToolUsageError as exc:
            errors.append(exc)
        return {"messages": []}

    await one_node(one_use(node)).ainvoke({"messages": []}, config())
    assert [str(e) for e in errors] == [approval.UNSUPPORTED]
    with pytest.raises(ToolUsageError):
        approval.resume_count({"configurable": {}})
