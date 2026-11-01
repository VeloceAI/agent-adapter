"""tool_handler: SQS batches with partial failure, EventBridge, direct events, time budgets."""

import asyncio
import json
import logging

import pytest

from agent_adapter import ApprovalDecision
from agent_adapter.errors import ToolFailed, ToolUsageError
from agent_adapter.handlers import RunSync, tool_handler
from agent_adapter.handlers.serverless import MIN_BUDGET_S, _budget


class Context:
    """A Lambda context stand-in with a fixed remaining time."""

    def __init__(self, remaining_ms: int) -> None:
        self.remaining_ms = remaining_ms

    def get_remaining_time_in_millis(self) -> int:
        return self.remaining_ms


class RecordingRunner(RunSync):
    """RunSync that records each timeout it was given."""

    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    def run(self, work, *, timeout=None):
        self.timeouts.append(timeout)
        return super().run(work, timeout=timeout)


def sqs(*bodies: object, fifo: bool = False) -> dict[str, object]:
    arn = "arn:aws:sqs:eu-west-1:1:tools" + (".fifo" if fifo else "")
    records = [
        {
            "messageId": f"m{i}",
            "body": body if isinstance(body, str) else json.dumps(body),
            "eventSource": "aws:sqs",
            "eventSourceARN": arn,
        }
        for i, body in enumerate(bodies)
    ]
    return {"Records": records}


ECHO = {"tool": "echo", "arguments": {"text": "hi"}}
BROKEN = {"tool": "broken", "arguments": {"token": "x"}}
FILING = {"tool": "file_return", "arguments": {"pan": "P", "year": 2026}}


def failures(reply: dict[str, object]) -> list[object]:
    return [item["itemIdentifier"] for item in reply["batchItemFailures"]]


def test_tool_handler_direct_event_returns_the_reply(executor):
    reply = tool_handler(executor)(ECHO, None)
    assert reply["status"] == "ok" and reply["result"]["text"] == "hi"


def test_tool_handler_direct_event_errors_raise_for_the_host_to_retry(executor):
    with pytest.raises(ToolFailed):
        tool_handler(executor)(BROKEN)


def test_tool_handler_eventbridge_uses_detail_and_a_stable_event_id(executor, store):
    event = {"id": "evt-1", "detail-type": "tool", "source": "app", "detail": FILING}
    handler = tool_handler(executor)
    first, again = handler(event), handler(event)
    assert first["status"] == "pending_approval"
    assert first["approval_id"] == again["approval_id"]


def test_tool_handler_sqs_batch_reports_only_failed_messages(executor):
    reply = tool_handler(executor)(sqs(ECHO, BROKEN, "not json", FILING, ECHO))
    assert failures(reply) == ["m1", "m2"]


def test_tool_handler_sqs_fifo_stops_at_the_first_failure(executor):
    reply = tool_handler(executor)(sqs(ECHO, BROKEN, ECHO, ECHO, fifo=True))
    assert failures(reply) == ["m1", "m2", "m3"]


def test_tool_handler_sqs_fifo_detected_by_message_group(executor):
    event = sqs(BROKEN, ECHO)
    event["Records"][0]["attributes"] = {"MessageGroupId": "g"}
    assert failures(tool_handler(executor)(event)) == ["m0", "m1"]


def test_tool_handler_sqs_resume_message_runs_the_decided_call(executor, tools, store):
    handler = tool_handler(executor)
    assert failures(handler(sqs(FILING))) == []
    (record,) = run_list(store)
    store_decide(store, record.approval_id)
    assert failures(handler(sqs({"approval_id": record.approval_id}))) == []
    assert tools.filed == [("P", 2026)]
    # a redelivered resume runs nothing and is acked (already done), not sent to the DLQ
    assert failures(handler(sqs({"approval_id": record.approval_id}))) == []
    assert tools.filed == [("P", 2026)]


def run_list(store):
    return asyncio.run(store.list())


def store_decide(store, approval_id):
    asyncio.run(store.decide(approval_id, ApprovalDecision(approved=True)))


def test_tool_handler_sqs_timeout_fails_unfinished_messages(tools, store, caplog):
    from agent_adapter import DeferredApproval, Executor

    @tools.registry.tool
    async def slow() -> str:
        """Takes too long."""
        await asyncio.sleep(30)
        return "late"

    executor = Executor(tools.registry, approval=DeferredApproval(store))
    handler = tool_handler(executor, timeout=1.0)
    reply = handler(sqs(ECHO, {"tool": "slow"}, ECHO))
    assert failures(reply) == ["m1", "m2"]
    assert "ran out of time" in caplog.text


def test_tool_handler_single_event_timeout_raises(tools, store, caplog):
    from agent_adapter import DeferredApproval, Executor

    @tools.registry.tool
    async def slow() -> str:
        """Takes too long."""
        await asyncio.sleep(30)
        return "late"

    executor = Executor(tools.registry, approval=DeferredApproval(store))
    with pytest.raises(TimeoutError):
        tool_handler(executor, timeout=0.5)({"tool": "slow"})
    assert "ran out of time" in caplog.text


def test_tool_handler_caps_the_budget_by_remaining_time(executor):
    runner = RecordingRunner()
    handler = tool_handler(executor, timeout=30, margin_s=1.0, runner=runner)
    handler(ECHO, Context(5_000))
    handler(ECHO, Context(120_000))
    handler(ECHO, None)
    assert runner.timeouts == [4.0, 30, 30]


def test_budget_never_drops_below_the_minimum():
    assert _budget(10, 1.0, Context(500)) == MIN_BUDGET_S


def test_tool_handler_tools_allowlist_applies_to_batches(executor, caplog):
    caplog.set_level(logging.WARNING)
    reply = tool_handler(executor, tools={"echo"})(sqs(ECHO, FILING))
    assert failures(reply) == ["m1"]
    assert "ToolNotFound" in caplog.text


@pytest.mark.parametrize(
    ("event", "match"),
    [
        ([1], "JSON object"),
        ({"Records": "x"}, "must be a list"),
        ({"Records": [{"eventSource": "aws:sns"}]}, "SQS batches only"),
        ({"Records": [{"eventSource": "aws:sqs", "body": "{}"}]}, "no messageId"),
        ({"Records": [{"eventSource": "aws:sqs", "messageId": "m"}]}, "no string body"),
    ],
)
def test_tool_handler_refuses_malformed_events_before_running(executor, event, match):
    with pytest.raises(ToolUsageError, match=match):
        tool_handler(executor)(event)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"timeout": 0}, "timeout"),
        ({"timeout": True}, "timeout"),
        ({"timeout": float("inf")}, "timeout"),
        ({"margin_s": -1}, "margin_s"),
    ],
)
def test_tool_handler_validates_options(executor, options, match):
    with pytest.raises(ToolUsageError, match=match):
        tool_handler(executor, **options)


def test_tool_handler_zero_margin_is_allowed(executor):
    assert tool_handler(executor, margin_s=0)(ECHO, Context(2_000))["status"] == "ok"
