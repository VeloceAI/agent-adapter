"""Durable flow runs: a fail-closed sink behind the step's slice sink stops calls; content flows."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.audit.errors import AuditUnavailable
from agent_adapter.core.errors import FlowError
from agent_adapter.core.types import Event
from agent_adapter.evals import cases_from_runs
from agent_adapter.exec import Executor
from agent_adapter.exec.checkpoints import MemoryCheckpointStore
from agent_adapter.flow import FlowRunner
from agent_adapter.integrations._durable_flows import DurableFlows
from agent_adapter.integrations._durable_slice import SliceSink
from agent_adapter.tracing.events import members
from agent_adapter.tracing.sinks import EventLog
from agent_adapter.viewer import Recorder
from tests.integrations.durable.kit import KEY, RAN, orders, registry


class LostRecords:
    """A fail-closed sink that has lost a record."""

    def emit(self, event: Event) -> None:
        pass

    def settled(self) -> bool:
        return True

    def require(self) -> None:
        raise AuditUnavailable(1)


def flows_over(executor: Executor) -> DurableFlows:
    runner = FlowRunner(executor, checkpoints=MemoryCheckpointStore(), checkpoint_key=KEY)
    return DurableFlows(runner, [orders()])


def test_the_slice_sink_shows_what_is_behind_it() -> None:
    inner = EventLog()
    assert list(members(SliceSink(inner, first=True)))[1:] == [inner]


async def test_a_lost_audit_record_stops_tool_calls_inside_a_durable_step() -> None:
    RAN.clear()
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    executor.add_sink(EventLog())
    executor.add_sink(LostRecords())
    with pytest.raises(FlowError) as failed:
        await flows_over(executor).advance("orders", "r1", {"order": "o-9"})
    assert isinstance(failed.value.__cause__, AuditUnavailable)
    assert RAN["charge"] == 0  # the side effect never ran


async def test_a_durable_run_records_content_and_becomes_a_case(tmp_path: Path) -> None:
    RAN.clear()
    recorder = Recorder(tmp_path / "runs", capture_content=True)
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    executor.add_sink(recorder)
    flows = flows_over(executor)
    report = await flows.advance("orders", "r2", {"order": "o-9"})
    while report.status == "running":
        report = await flows.advance("orders", "r2")
    assert report.status == "done"
    assert recorder.flush()
    recorder.stop()
    (case,) = cases_from_runs(tmp_path / "runs").cases
    assert case.case.input == {"order": "o-9", "note": "", "sent": ""}
    assert case.case.expected == {"order": "o-9", "note": "charged o-9", "sent": "shipped o-9"}
