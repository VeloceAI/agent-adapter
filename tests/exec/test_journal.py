"""The replay journal keeps only sealable JSON, nests child journals, and settles own stores."""

import json
import logging
from typing import Annotated

import pytest
from pydantic import BaseModel

from agent_adapter import ApprovalDecision, ApprovalRequest, Registry
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import ApprovalAlreadyDecided, ToolError, ToolUsageError
from agent_adapter.core.types import ErrorInfo, ToolCall, ToolResult
from agent_adapter.exec import Executor
from agent_adapter.exec._approve import _settle_own, approval_store
from agent_adapter.exec._journal import _make_journal, unsealable_calls
from agent_adapter.exec._resume import _visit_scope

YES = ApprovalDecision(approved=True)
registry = Registry()
RAN: list[str] = []


class Opaque:
    pass


class Hit(BaseModel):
    text: str = ""


@registry.tool
async def opaque() -> object:
    """Returns a value with no canonical form."""
    RAN.append("opaque")
    return Opaque()


@registry.tool
async def crash() -> str:
    """Always fails."""
    RAN.append("crash")
    raise ToolError("no", tool="crash")


@registry.tool
async def read(key: str) -> str:
    """Read a value."""
    RAN.append(f"read {key}")
    return f"value of {key}"


@pytest.fixture(autouse=True)
def _reset():
    RAN.clear()


async def test_an_unsealable_result_is_marked_as_run_and_never_runs_again():
    executor = Executor(registry)
    journal = _make_journal()
    with _visit_scope(None, journal):
        await executor.run("opaque", {}, call_id="o1")
    data = journal.data()
    assert data["calls"][0][3]["unsealable"] == "ran, result not sealable"
    assert data["calls"][0][3]["call_id"] == "o1"  # the call it stands for
    [problem] = unsealable_calls(data)
    assert "opaque" in problem and "occurrence 0" in problem
    with _visit_scope(None, _make_journal(data)), pytest.raises(ToolUsageError, match="opaque"):
        await executor.run("opaque", {})
    assert RAN == ["opaque"]


async def test_a_call_whose_arguments_have_no_canonical_form_is_marked():
    journal = _make_journal()
    slot = journal.claim("read", {"key": object()})
    journal.record(slot, _ok("v"))
    [problem] = unsealable_calls(journal.data())
    assert "read" in problem and "arguments" in problem
    nested = _make_journal()
    child = nested.child("item:0")
    child.record(child.claim("opaque", {}), _ok(object()))
    assert unsealable_calls(nested.data()) and unsealable_calls({}) == []


async def test_a_custom_result_is_kept_only_with_a_decoder_too():
    executor = Executor(registry)
    encoded = _make_journal(encoder=lambda v: "opaque")
    with _visit_scope(None, encoded):
        await executor.run("opaque", {})
    assert unsealable_calls(encoded.data())  # no decoder: it could not be replayed
    both = _make_journal(encoder=lambda v: "opaque", decoder=lambda name, v: Opaque())
    with _visit_scope(None, both):
        await executor.run("opaque", {})
    again = _make_journal(both.data(), encoder=lambda v: "opaque", decoder=lambda n, v: Opaque())
    with _visit_scope(None, again):
        assert isinstance(await executor.run("opaque", {}), Opaque)
    assert RAN == ["opaque", "opaque"]


async def test_invalid_arguments_are_keyed_as_given():
    executor = Executor(registry)
    journal = _make_journal()
    with _visit_scope(None, journal):
        result = await executor.run_call(_call("read", {"key": ["not", "text"]}))
        assert journal.claim("read", {"key": ["not", "text"]}).occurrence == 1
    assert result.status == "error" and RAN == []


def test_a_result_class_its_tool_does_not_annotate_is_marked():
    journal = _make_journal()
    journal.record(journal.claim("read", {"key": "k"}, returns=str), _ok(Hit()))
    assert "Hit" in unsealable_calls(journal.data())[0]
    typed = _make_journal()
    typed.record(typed.claim("read", {"key": "k"}, returns=list[Hit]), _ok([Hit()]))
    replay = _make_journal(typed.data()).claim("read", {"key": "k"}, returns=list[Hit])
    assert replay.result.value == [Hit()]
    meta = Annotated[Hit, {"unhashable": []}]
    assert _make_journal().claim("read", {"key": "k"}, returns=meta).allowed


def test_a_closed_journal_keeps_no_marker():
    journal = _make_journal()
    slot = journal.claim("read", {"key": object()})
    journal.close()
    journal.record(slot, _ok("v"))
    assert journal.data() == {}


def test_child_journals_nest_by_name_and_occurrence_and_close_with_the_parent():
    root = _make_journal(calls=False)
    assert root.claim("read", {"key": "k"}) is None  # a Map visit records no calls itself
    first, second = root.child("item:0"), root.child("item:0")
    first.record(first.claim("read", {"key": "a"}), _ok("a"))
    second.record(second.claim("read", {"key": "b"}), _ok("b"))
    data = root.data()
    assert [c[:2] for c in data["children"]] == [["item:0", 0], ["item:0", 1]]
    assert root.data(drop=["item:0"]) == {}
    reloaded = _make_journal(json.loads(json.dumps(data)))
    replay = reloaded.child("item:0").claim("read", {"key": "a"})
    assert replay is not None and replay.result.value == "a"
    root.close()
    late = root.child("item:9")
    assert late.claim("read", {"key": "z"}) is None


def test_a_journaled_error_with_an_unknown_class_runs_again(caplog):
    journal = _make_journal()
    slot = journal.claim("crash", {})
    error = ToolError("no", tool="crash")
    journal.record(
        slot,
        ToolResult(
            call_id="c",
            name="crash",
            status="error",
            error=ErrorInfo.from_exception(error),
            exception=error,
        ),
    )
    data = journal.data()
    data["calls"][0][3]["error"]["exception"] = "nowhere.Gone"
    with caplog.at_level(logging.WARNING):
        assert _make_journal(data).claim("crash", {}).result is None
    assert "cannot be replayed" in caplog.text


def test_foreign_exceptions_are_marked_unsealable():
    journal = _make_journal()
    slot = journal.claim("crash", {})
    error = RuntimeError("x")
    journal.record(
        slot,
        ToolResult(
            call_id="c",
            name="crash",
            status="error",
            error=ErrorInfo.from_exception(error),
            exception=error,
        ),
    )
    assert "AdapterError" in unsealable_calls(journal.data())[0]


async def test_settle_own_skips_another_calls_record_and_tolerates_a_racing_decide():
    store = MemoryApprovalStore()
    record = await store.create(_request("call_a"))
    await _settle_own(store, _request("call_b", approval_id=record.approval_id), YES)
    assert (await store.get(record.approval_id)).status == "pending"  # not ours: untouched

    class Racing(MemoryApprovalStore):
        async def decide(self, approval_id, decision):
            await super().decide(approval_id, decision)
            raise ApprovalAlreadyDecided(approval_id, status="approved")

    racing = Racing()
    raced = await racing.create(_request("call_c"))
    await _settle_own(racing, _request("call_c", approval_id=raced.approval_id), YES)
    assert (await racing.get(raced.approval_id)).consumed
    with pytest.raises(ApprovalAlreadyDecided):
        await _settle_own(
            racing,
            _request("call_c", approval_id=raced.approval_id),
            ApprovalDecision(approved=False),
        )


async def test_settle_own_refuses_when_another_caller_consumed_the_record_first():
    class Raced(MemoryApprovalStore):
        async def consume(self, approval_id):
            await super().consume(approval_id)
            return False

    store = Raced()
    record = await store.create(_request("call_d"))
    with pytest.raises(ApprovalAlreadyDecided):
        await _settle_own(store, _request("call_d", approval_id=record.approval_id), YES)


def test_a_port_store_is_found_only_when_it_is_an_approval_store():
    store = MemoryApprovalStore()
    assert approval_store(DeferredApproval(store)) is store
    assert approval_store(object()) is None and approval_store(None) is None


def _request(call_id, *, approval_id="apr_x"):
    return ApprovalRequest(
        approval_id=approval_id,
        run_id="run_1",
        call_id=call_id,
        tool="read",
        arguments={"key": "k"},
    )


def _call(name, arguments):
    return ToolCall(name=name, arguments=arguments)


def _ok(value):
    return ToolResult(call_id="c", name="read", status="ok", value=value)
