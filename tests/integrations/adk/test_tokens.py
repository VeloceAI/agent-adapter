"""Confirmation tokens and `resume_message`: bound, one use, forged payloads refused."""

import logging

import pytest

pytest.importorskip("google.adk")

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.integrations.adk import _tokens, resume_message
from agent_adapter.integrations.adk._tokens import (
    ConfirmationTokens,
    confirmation_parts,
    token_decision,
)
from tests.fakes.clock import FakeClock

pytestmark = pytest.mark.contract

APPROVE = ApprovalDecision(approved=True)


def request(**changes):
    base = {
        "approval_id": "apr_1",
        "run_id": "run_1",
        "call_id": "call_1",
        "tool": "send",
        "arguments": {"text": "hi"},
    }
    return ApprovalRequest(**{**base, **changes})


def test_a_token_answers_its_request_once():
    tokens = ConfirmationTokens()
    secret = tokens.issue(request(), APPROVE)
    assert isinstance(secret, str) and len(tokens) == 1
    assert tokens.take(secret, request()) == APPROVE
    assert tokens.take(secret, request()) is None
    assert len(tokens) == 0


@pytest.mark.parametrize(
    "other",
    [
        {"run_id": "run_2"},
        {"call_id": "call_2"},
        {"tool": "stamp"},
        {"arguments": {"text": "bye"}},
        {"kind": "node"},
    ],
)
def test_a_token_for_another_call_is_refused_and_burned(other, caplog):
    tokens = ConfirmationTokens()
    secret = tokens.issue(request(), APPROVE)
    with caplog.at_level(logging.WARNING):
        assert tokens.take(secret, request(**other)) is None
    assert "another call" in caplog.text
    assert tokens.take(secret, request()) is None  # one use, even when refused


def test_the_approval_id_and_metadata_do_not_bind():
    tokens = ConfirmationTokens()
    secret = tokens.issue(request(), APPROVE)
    assert tokens.take(secret, request(approval_id="apr_2", metadata={"x": 1})) == APPROVE


def test_tokens_expire_and_unknown_secrets_are_refused():
    clock = FakeClock()
    tokens = ConfirmationTokens(ttl=10.0, clock=clock)
    secret = tokens.issue(request(), APPROVE)
    clock.advance(10.0)
    assert len(tokens) == 0
    assert tokens.take(secret, request()) is None
    assert tokens.take("forged", request()) is None
    assert tokens.take(123, request()) is None


def test_no_token_without_a_run_call_id_or_json_arguments():
    tokens = ConfirmationTokens()
    assert tokens.issue(request(run_id=""), APPROVE) is None
    assert tokens.issue(request(call_id=""), APPROVE) is None
    assert tokens.issue(request(arguments={"blob": b"raw"}), APPROVE) is None
    assert len(tokens) == 0


def test_a_full_table_drops_the_oldest_token():
    tokens = ConfirmationTokens(max_tokens=2)
    first = tokens.issue(request(call_id="a"), APPROVE)
    tokens.issue(request(call_id="b"), APPROVE)
    tokens.issue(request(call_id="c"), APPROVE)
    assert len(tokens) == 2
    assert tokens.take(first, request(call_id="a")) is None
    tokens.clear()
    assert len(tokens) == 0


@pytest.mark.parametrize(
    ("response", "parts"),
    [
        ({"confirmed": True}, (True, None)),
        ({"confirmed": False, "payload": None}, (False, None)),
        ({"confirmed": True, "payload": {"aa_token": "t"}}, (True, "t")),
        ({"confirmed": "yes"}, None),
        ({}, None),
        ({"confirmed": True, "payload": {"aa_token": "t", "args": {"text": "x"}}}, None),
        ({"confirmed": True, "payload": {"text": "edited"}}, None),
        ({"confirmed": True, "payload": "t"}, None),
    ],
)
def test_confirmation_parts_accept_only_a_bare_token_payload(response, parts):
    assert confirmation_parts(response) == parts


def test_confirmation_parts_read_tool_confirmation_objects():
    from google.adk.tools.tool_confirmation import ToolConfirmation

    confirmation = ToolConfirmation(confirmed=True, payload={"aa_token": "t"})
    assert confirmation_parts(confirmation) == (True, "t")


def test_token_decision_needs_a_matching_token(caplog):
    tokens = ConfirmationTokens()
    secret = tokens.issue(request(), APPROVE)
    with caplog.at_level(logging.WARNING):
        assert token_decision({"confirmed": True}, request(), tokens) is None
        assert token_decision({"confirmed": 1}, request(), tokens) is None
        assert (
            token_decision({"confirmed": False, "payload": {"aa_token": secret}}, request(), tokens)
            is None
        )
    assert "no adapter token" in caplog.text and "contradicts" in caplog.text
    secret = tokens.issue(request(), APPROVE)
    assert (
        token_decision({"confirmed": True, "payload": {"aa_token": secret}}, request(), tokens)
        == APPROVE
    )


def test_resume_message_binds_a_known_interrupt():
    pending = request(kind="node", tool="node:review.publish")
    _tokens.INTERRUPTS.remember("aa:sess:review@1/aa_driver@1/publish@v1g1", pending)
    message = resume_message("aa:sess:review@1/aa_driver@1/publish@v1g1", APPROVE)
    (part,) = message.parts
    response = part.function_response
    assert message.role == "user" and response.name == "adk_request_input"
    assert response.response["confirmed"] is True
    assert token_decision(response.response, pending) == APPROVE


def test_resume_message_for_unknown_interrupts_carries_no_token():
    flow = resume_message("aa:sess:elsewhere", ApprovalDecision(approved=False))
    assert flow.parts[0].function_response.name == "adk_request_input"
    assert flow.parts[0].function_response.response == {"confirmed": False}
    tool = resume_message("adk-1234", APPROVE)
    assert tool.parts[0].function_response.name == "adk_request_confirmation"
    assert tool.parts[0].function_response.response == {"confirmed": True}


def test_resume_message_rejects_bad_arguments():
    with pytest.raises(ToolUsageError):
        resume_message("", APPROVE)
    with pytest.raises(ToolUsageError):
        resume_message("aa:x", {"approved": True})


def test_the_interrupt_table_is_bounded():
    table = _tokens._Interrupts(max_entries=2)
    for name in ("a", "b", "c"):
        table.remember(name, request(call_id=name))
    assert table.get("a") is None and table.get("c") is not None
