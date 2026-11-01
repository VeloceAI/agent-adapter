"""handle_envelope: runs, parks, resumes once, refuses what a queue must not run, logs safely."""

import logging

import pytest

from agent_adapter import ApprovalDecision, ApprovalRequest, Executor
from agent_adapter.errors import (
    ApprovalDenied,
    ApprovalNotFound,
    ToolFailed,
    ToolNotFound,
    ToolUsageError,
)
from agent_adapter.handlers import Envelope, handle_envelope
from tests.handlers.conftest import SECRET

FILING = {"tool": "file_return", "arguments": {"pan": "ABCDE1234F", "year": 2026}}


async def test_handle_envelope_ok_reply_is_json_safe(executor):
    payload = {"tool": "echo", "arguments": {"text": "hi"}, "run_id": "r1", "call_id": "c1"}
    reply = await handle_envelope(executor, payload)
    assert reply == {
        "status": "ok",
        "tool": "echo",
        "run_id": "r1",
        "call_id": "c1",
        "result": {"text": "hi", "pair": [1, 2]},
    }


async def test_handle_envelope_accepts_a_parsed_envelope(executor):
    reply = await handle_envelope(executor, Envelope("echo", {"text": "x"}, "r", "c"))
    assert reply["result"]["text"] == "x"


async def test_handle_envelope_redelivery_parks_the_same_approval(executor, tools, store):
    first = await handle_envelope(executor, FILING, message_id="m-1", source="sqs")
    again = await handle_envelope(executor, FILING, message_id="m-1", source="sqs")
    assert first["status"] == again["status"] == "pending_approval"
    assert first["approval_id"] == again["approval_id"]
    assert len(await store.list()) == 1 and tools.filed == []


async def test_handle_envelope_resume_runs_a_decided_call_once(executor, tools, store):
    parked = await handle_envelope(executor, FILING, message_id="m-1")
    approval_id = parked["approval_id"]
    still = await handle_envelope(executor, {"approval_id": approval_id})
    assert still == parked
    assert tools.filed == []

    edited = ApprovalDecision(approved=True, edited_args={"pan": "ABCDE1234F", "year": 2025})
    await store.decide(approval_id, edited)
    done = await handle_envelope(executor, {"approval_id": approval_id})
    assert done["status"] == "ok" and done["result"] == "filed ABCDE1234F 2025"
    assert (done["run_id"], done["call_id"]) == (parked["run_id"], parked["call_id"])
    again = await handle_envelope(executor, {"approval_id": approval_id})  # redelivered
    assert again == {"status": "already_done", "approval_id": approval_id, **_ids(parked)}
    assert tools.filed == [("ABCDE1234F", 2025)]


def _ids(reply: dict[str, object]) -> dict[str, object]:
    return {key: reply[key] for key in ("tool", "run_id", "call_id")}


async def test_handle_envelope_resume_of_a_denied_call_raises(executor, tools, store):
    parked = await handle_envelope(executor, FILING)
    await store.decide(parked["approval_id"], ApprovalDecision(approved=False, reason="no"))
    with pytest.raises(ApprovalDenied):
        await handle_envelope(executor, {"approval_id": parked["approval_id"]})
    assert tools.filed == []


async def test_handle_envelope_resume_of_an_unknown_approval(executor):
    with pytest.raises(ApprovalNotFound):
        await handle_envelope(executor, {"approval_id": "apr_missing"})


async def test_handle_envelope_resume_needs_deferred_approval(tools):
    with pytest.raises(ToolUsageError, match="DeferredApproval"):
        await handle_envelope(Executor(tools.registry), {"approval_id": "apr_1"})


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"kind": "node", "tool": "node:flow.gate"}, "flow node"),
        ({"metadata": {"principal": "alice", "principal_scopes": []}}, "HTTP tool server"),
        ({"metadata": {"anonymous": True}}, "HTTP tool server"),
        ({"metadata": {"origin": "http"}}, "HTTP tool server"),
    ],
)
async def test_handle_envelope_refuses_records_other_surfaces_resume(
    executor, tools, store, changes, match
):
    fields = {
        "approval_id": "",
        "run_id": "run_x",
        "call_id": "call_x",
        "tool": "file_return",
        "arguments": FILING["arguments"],
        **changes,
    }
    record = await store.create(ApprovalRequest(**fields))
    await store.decide(record.approval_id, ApprovalDecision(approved=True))
    with pytest.raises(ToolUsageError, match=match):
        await handle_envelope(executor, {"approval_id": record.approval_id})
    assert tools.filed == []


async def test_handle_envelope_tools_allowlist(executor, store):
    with pytest.raises(ToolNotFound, match="not served by this handler"):
        await handle_envelope(executor, FILING, tools={"echo"})
    echo = {"tool": "echo", "arguments": {"text": "a"}}
    assert (await handle_envelope(executor, echo, tools=["echo"]))["status"] == "ok"
    assert await store.list() == []


async def test_handle_envelope_failure_logs_kind_not_secrets(executor, caplog):
    caplog.set_level(logging.DEBUG, logger="agent_adapter")
    with pytest.raises(ToolFailed):
        await handle_envelope(
            executor, {"tool": "broken", "arguments": {"token": SECRET}}, message_id="m-9"
        )
    ours = [r for r in caplog.records if r.name.startswith("agent_adapter.handlers")]
    warnings = [r for r in ours if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "m-9" in warnings[0].getMessage() and "ToolFailed" in warnings[0].getMessage()
    assert all(SECRET not in r.getMessage() for r in caplog.records if r.levelno >= logging.INFO)


async def test_handle_envelope_ok_logs_ids_not_arguments_or_results(executor, caplog):
    caplog.set_level(logging.INFO, logger="agent_adapter.handlers")
    await handle_envelope(executor, {"tool": "echo", "arguments": {"text": SECRET}})
    messages = [r.getMessage() for r in caplog.records if r.name.startswith("agent_adapter.h")]
    assert any("envelope ok" in m for m in messages)
    assert all(SECRET not in m for m in messages)


async def test_handle_envelope_bad_envelope_is_logged_and_raised(executor, caplog):
    with pytest.raises(ToolUsageError):
        await handle_envelope(executor, {"tool": "echo", "approved": True})
    assert "ToolUsageError" in caplog.text


async def test_handle_envelope_foreign_errors_log_at_error_and_raise(executor, store, caplog):
    async def down(approval_id: str) -> None:
        raise ConnectionError(f"store down {SECRET}")

    store.get = down
    with pytest.raises(ConnectionError):
        await handle_envelope(executor, {"approval_id": "apr_1"})
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and SECRET not in errors[0].getMessage()
