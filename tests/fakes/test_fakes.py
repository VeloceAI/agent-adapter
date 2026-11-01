import pytest

from agent_adapter.core.ports import ApprovalPort, LedgerPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest, LedgerEntry
from agent_adapter.hooks import ShortCircuit
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.fakes.cache import BrokenCache, DictCache
from tests.fakes.clock import FakeClock
from tests.fakes.hooks import RaisingHook, RecordingHook
from tests.fakes.sinks import RecordingLedger

REQUEST = ApprovalRequest(approval_id="apr_1", run_id="r", call_id="c", tool="t")


async def test_fake_clock_moves_only_when_told():
    clock = FakeClock(10.0)
    assert clock() == clock.now() == 10.0
    clock.advance(1.5)
    await clock.sleep(2)
    assert clock() == 13.5 and clock.sleeps == [2]


async def test_dict_cache_ttl_and_delete():
    clock = FakeClock()
    cache = DictCache(clock)
    await cache.set("a", b"1", ttl=5)
    await cache.set("b", b"2")
    assert await cache.get("a") == b"1"
    clock.advance(5)
    assert await cache.get("a") is None and await cache.get("b") == b"2"
    await cache.delete("b")
    assert await cache.get("b") is None


async def test_broken_cache_raises_everywhere():
    cache = BrokenCache()
    for call in (cache.get("k"), cache.set("k", b""), cache.delete("k")):
        with pytest.raises(ConnectionError):
            await call
    assert cache.calls == 3


def test_recording_ledger_is_idempotent_and_filters():
    ledger = RecordingLedger()
    assert isinstance(ledger, LedgerPort)
    a = LedgerEntry(run_id="r1", call_id="c1", kind="llm", cost_usd=1.0, thread_id="t1")
    b = LedgerEntry(run_id="r2", call_id="c2", kind="llm", cost_usd=2.0, thread_id="t1")
    c = LedgerEntry(run_id="r2", call_id="c3", kind="tool", thread_id="t2")
    for entry in (a, a, b, c):
        ledger.record(entry)
    assert ledger.records == 4 and len(ledger.entries()) == 3
    assert ledger.total_usd() == 3.0 and ledger.total_usd("r1") == 1.0
    assert ledger.total_usd(thread_id="t1") == 3.0
    assert ledger.entries("r2", thread_id="t2") == [c]


async def test_scripted_approval_returns_and_raises_in_order():
    port = ScriptedApproval([ApprovalDecision(approved=True), PermissionError("no")])
    assert isinstance(port, ApprovalPort)
    assert (await port.request(REQUEST)).approved
    with pytest.raises(PermissionError):
        await port.request(REQUEST)
    with pytest.raises(AssertionError, match="no scripted decision left"):
        await port.request(REQUEST)
    assert port.requests == [REQUEST] * 3


async def test_recording_approval_wraps_a_port():
    decision = ApprovalDecision(approved=False, reason="risky")
    port = RecordingApproval(ScriptedApproval([decision]))
    assert await port.request(REQUEST) == decision
    assert port.requests == [REQUEST] and port.decisions == [decision]


async def test_hook_fakes():
    hook = RecordingHook({"before_tool": ShortCircuit(1)})
    assert await hook.before_tool("inv") == ShortCircuit(1)
    assert await hook.after_llm("inv", "resp") is None
    assert hook.phases() == ["before_tool", "after_llm"]
    raising = RaisingHook(ValueError("x"), phases=["after_tool"])
    assert await raising.before_tool("inv") is None
    with pytest.raises(ValueError):
        await raising.after_tool("inv", "result")
    assert raising.phases() == ["before_tool", "after_tool"]
