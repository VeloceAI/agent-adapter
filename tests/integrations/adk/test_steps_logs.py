"""ADK flow steps log a failed approval settle by exception type only above DEBUG."""

import logging

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.integrations.adk import _steps

SECRET = "amount 9001 over the card limit"
REQUEST = ApprovalRequest(
    approval_id="apr_1", run_id="run_1", call_id="call_1", tool="pay", arguments={}
)


class _Store:
    async def get(self, approval_id):
        return object()


class _Port:
    async def prepare(self, request):
        raise AssertionError("not used")

    async def consume(self, record, request):
        raise ToolUsageError(SECRET)


async def _noop(*args):
    return None


def _above_debug(caplog):
    return [r for r in caplog.records if r.levelno > logging.DEBUG and r.name == _steps.__name__]


async def test_settled_failure_logs_the_type_not_the_text(monkeypatch, caplog):
    async def refuse(*args):
        raise ToolUsageError(SECRET)

    monkeypatch.setattr(_steps, "spend_once", refuse)
    caplog.set_level(logging.DEBUG, logger=_steps.__name__)

    assert await _steps._settled(ApprovalDecision(approved=True), REQUEST, object()) is None

    [record] = _above_debug(caplog)
    assert record.levelno == logging.WARNING
    assert "ToolUsageError" in record.getMessage()
    assert SECRET not in record.getMessage()


async def test_stored_decision_failure_logs_the_type_not_the_text(monkeypatch, caplog):
    monkeypatch.setattr(_steps, "_refuse_spent", _noop)
    monkeypatch.setattr(_steps, "approval_store", lambda port: _Store())
    caplog.set_level(logging.DEBUG, logger=_steps.__name__)

    assert await _steps._stored_decision(REQUEST, _Port()) is None

    [record] = _above_debug(caplog)
    assert record.levelno == logging.WARNING
    assert "ToolUsageError" in record.getMessage()
    assert SECRET not in record.getMessage()
    assert any(SECRET in str(r.exc_info[1]) for r in caplog.records if r.exc_info)
