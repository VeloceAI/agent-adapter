"""The approval digest that binds a record to its tool and original arguments."""

from typing import Any

import pytest
from pydantic import ValidationError

import agent_adapter.approval as approval
from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.store import ApprovalRecord, MemoryApprovalStore, request_digest
from agent_adapter.core.errors import ApprovalPending, ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest

YES = ApprovalDecision(approved=True, decided_by="ana")


def req(arguments: dict[str, Any] | None = None, *, tool: str = "pay", kind: str = "tool"):
    return ApprovalRequest(
        approval_id="apr_1",
        run_id="run_1",
        call_id="c1",
        tool=tool,
        kind=kind,
        arguments={} if arguments is None else arguments,
    )


def test_digest_is_canonical_and_covers_kind_tool_and_arguments():
    args = {"a": 1, "b": [1, {"c": "é"}], "f": 1.5, "n": None, "t": True}
    base = request_digest(req(args))
    assert base == request_digest(req(dict(reversed(list(args.items())))))
    assert len(base) == 64
    assert base != request_digest(req({**args, "a": 2}))
    assert base != request_digest(req(args, tool="refund"))
    assert base != request_digest(req(args, kind="node"))


def test_emoji_and_its_surrogate_pair_stay_distinct():
    emoji = request_digest(req({"s": chr(0x1F600)}))
    pair = request_digest(req({"s": chr(0xD83D) + chr(0xDE00)}))  # the UTF-16 pair
    assert emoji != pair


async def test_a_lone_surrogate_request_is_stored_and_matched():
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    lone = {"note": "x" + chr(0xD800) + "y"}
    with pytest.raises(ApprovalPending):
        await port.request(req(lone))
    await store.decide("apr_1", YES)
    with pytest.raises(ToolUsageError, match="reused"):
        await port.request(req({"note": "x" + chr(0xD801) + "y"}))
    assert (await port.request(req(lone))).approved


class Opaque:
    pass


@pytest.mark.parametrize(
    ("arguments", "where"),
    [
        ({"blob": b"\x00"}, "$.blob: bytes"),
        ({"pair": (1, 2)}, "$.pair: tuple"),
        ({"tags": {"a"}}, "$.tags: set"),
        ({"obj": Opaque()}, "$.obj: Opaque"),
        ({"opts": {1: "a", "b": 2}}, "$.opts: non-string key int"),
        ({"deep": [{"x": float("nan")}]}, "$.deep[0].x: non-finite float"),
    ],
)
async def test_non_json_arguments_fail_closed_at_request_time(arguments, where):
    store = MemoryApprovalStore()
    with pytest.raises(ToolUsageError, match="not JSON") as info:
        await DeferredApproval(store).request(req(arguments))
    assert where in str(info.value)
    assert await store.list() == []
    with pytest.raises(ToolUsageError, match="not JSON"):
        request_digest(req(arguments))


def test_a_record_always_computes_its_digest_from_the_request():
    record = ApprovalRecord(request=req({"a": 1}), status="pending")
    assert record.args_digest == request_digest(req({"a": 1}))
    same = ApprovalRecord(request=req({"a": 1}), status="pending", args_digest=record.args_digest)
    assert same == record
    loaded = ApprovalRecord.model_validate(  # e.g. a row read back from a database
        {"request": req({"a": 1}).model_dump(), "status": "approved"}
    )
    assert loaded.args_digest == record.args_digest
    assert ApprovalRecord.model_validate(loaded) is loaded  # instances pass through


def test_a_record_without_a_request_is_a_validation_error():
    with pytest.raises(ValidationError):
        ApprovalRecord.model_validate({"status": "pending"})


def test_a_disagreeing_stored_digest_is_refused():
    with pytest.raises(ToolUsageError, match="args_digest does not match"):
        ApprovalRecord(request=req({"a": 1}), status="approved", args_digest="0" * 64)
    row = {"request": req({"a": 1}).model_dump(), "status": "approved", "args_digest": "f" * 64}
    with pytest.raises(ToolUsageError, match="args_digest does not match"):
        ApprovalRecord.model_validate(row)


def test_model_copy_with_a_new_request_recomputes_the_digest():
    record = ApprovalRecord(request=req({"a": 1}), status="pending")
    moved = record.model_copy(update={"request": req({"a": 2})})
    assert moved.args_digest == request_digest(req({"a": 2}))
    assert record.model_copy(update={"status": "approved"}).args_digest == record.args_digest
    with pytest.raises(ToolUsageError, match="args_digest does not match"):
        record.model_copy(update={"args_digest": "0" * 64})


class RowStore(MemoryApprovalStore):
    """Returns records rebuilt from tampered rows, like a database store would."""

    def __init__(self, row: dict[str, Any]) -> None:
        super().__init__()
        self.row = row

    async def find(self, run_id, call_id):
        return ApprovalRecord.model_validate(self.row)


async def test_a_tampered_store_row_fails_closed():
    # Approved for {"amount": 1}; the row's request was then edited to {"amount": 9000}.
    approved = ApprovalRecord(request=req({"amount": 1}), status="approved", decision=YES)
    row = approved.model_dump()
    row["request"]["arguments"] = {"amount": 9000}
    with pytest.raises(ToolUsageError, match="args_digest does not match"):
        await DeferredApproval(RowStore(row)).request(req({"amount": 9000}))


class ConstructStore(MemoryApprovalStore):
    """Returns one fixed record from `find`."""

    def __init__(self, record: ApprovalRecord) -> None:
        super().__init__()
        self.record = record

    async def find(self, run_id, call_id):
        return self.record


async def test_the_stored_request_wins_over_a_stale_digest():
    # model_construct skips validators: the digest claims {"amount": 9000}, the request is 1.
    stale = ApprovalRecord.model_construct(
        request=req({"amount": 1}),
        status="approved",
        decision=YES,
        consumed=False,
        args_digest=request_digest(req({"amount": 9000})),
    )
    with pytest.raises(ToolUsageError, match="reused"):
        await DeferredApproval(ConstructStore(stale)).request(req({"amount": 9000}))


def test_only_request_digest_is_public():
    assert "request_digest" in approval.__all__
    assert "approval_digest" not in approval.__all__
    assert not hasattr(approval, "approval_digest")
