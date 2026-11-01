"""A resumed node replays the executor calls it already finished instead of running them."""

import pytest

from agent_adapter import Registry
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor, RunLimits
from agent_adapter.flow import Flow, FlowRunner, Map
from tests.fakes.sinks import RecordingLedger
from tests.flow.helpers import State, resume_until_done

APPROVE = ApprovalDecision(approved=True)
registry = Registry()
RAN: list[str] = []


@registry.tool(confirm=True)
async def pay(amount: int) -> str:
    """Pay an amount (needs approval)."""
    RAN.append(f"pay{amount}")
    return f"paid {amount}"


@registry.tool(confirm=True)
async def nuke() -> str:
    """Delete everything (needs approval)."""
    RAN.append("nuke")
    return "nuked"


@registry.tool(once_per_run=True)
async def read(key: str) -> str:
    """Read a value once per run."""
    RAN.append(f"read {key}")
    return f"value of {key}"


@registry.tool
async def xboom(n: int) -> str:
    """Always fails."""
    RAN.append(f"xboom{n}")
    raise ConnectionError("down")


@pytest.fixture(autouse=True)
def _reset():
    RAN.clear()


def make(**options):
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()), **options)
    return FlowRunner(executor), executor


def code_node(executor_box, calls):
    async def go(state, ctx):
        values = [await executor_box[0].run(name, args, ctx) for name, args in calls]
        return {"note": " | ".join(values)}

    return Flow("replay", state=State).node("go", go).end("go")


async def test_two_confirm_calls_each_run_once_over_two_resumes():
    run, executor = make()
    flow = code_node([executor], [("pay", {"amount": 1}), ("nuke", {})])
    done, resumes = await resume_until_done(run, flow, await run.run(flow, {}))
    assert resumes == 2 and done.state.note == "paid 1 | nuked"
    assert RAN == ["pay1", "nuke"]


async def test_a_once_per_run_read_before_a_confirm_call_is_not_counted_again():
    run, executor = make()
    flow = code_node([executor], [("read", {"key": "k"}), ("pay", {"amount": 2})])
    done, _ = await resume_until_done(run, flow, await run.run(flow, {}))
    assert done.state.note == "value of k | paid 2"
    assert RAN == ["read k", "pay2"]


async def test_replayed_calls_do_not_count_against_max_calls():
    run, executor = make(limits=RunLimits(max_calls=2))
    flow = code_node([executor], [("read", {"key": "k"}), ("pay", {"amount": 3})])
    done, _ = await resume_until_done(run, flow, await run.run(flow, {}))
    assert done.status == "done" and RAN == ["read k", "pay3"]


async def test_ledger_entries_match_the_real_calls():
    ledger = RecordingLedger()
    run, executor = make(ledger=ledger)
    flow = code_node([executor], [("read", {"key": "k"}), ("pay", {"amount": 1}), ("nuke", {})])
    paused = await run.run(flow, {})
    run_id = paused.pending.run_id
    done, _ = await resume_until_done(run, flow, paused)
    assert done.status == "done"
    tool_entries = [e for e in ledger.entries(run_id) if e.kind == "tool"]
    assert len(tool_entries) == len(RAN) == 3


async def test_a_skipped_map_item_is_not_run_again_after_a_resume():
    run, _ = make()
    flow = (
        Flow("skips", state=State)
        .node("go", Map("items", tool_from="tool", into="log", on_error="skip"))
        .end("go")
    )
    items = [{"tool": "xboom", "n": 1}, {"tool": "pay", "amount": 4}]
    done, _ = await resume_until_done(run, flow, await run.run(flow, {"items": items}))
    assert done.state.log == ["paid 4"]
    assert RAN.count("xboom1") == 1 and RAN.count("pay4") == 1
