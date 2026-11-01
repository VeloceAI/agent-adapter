"""A failing gate port is logged by exception type only above DEBUG."""

import logging

import pytest

from agent_adapter.core.errors import FlowError, ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.integrations.langgraph import _gate

SECRET = "reviewer note: card 4111 is over its limit"


class _Failing:
    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        raise ToolUsageError(SECRET)


async def test_a_failing_gate_port_logs_the_type_not_the_text(caplog):
    request = ApprovalRequest(
        approval_id="a", run_id="r", call_id="c", tool="node:f.n", kind="node"
    )
    caplog.set_level(logging.DEBUG, logger=_gate.__name__)

    with pytest.raises(FlowError):
        await _gate.ask_gate("f", "n", request, _Failing(), peek=False)

    [record] = [r for r in caplog.records if r.levelno > logging.DEBUG]
    assert record.levelno == logging.WARNING
    assert "ToolUsageError" in record.getMessage()
    assert SECRET not in record.getMessage()
