"""A pending result from a store-backed port pickles and deep-copies (no store rides along)."""

import copy
import pickle

import pytest

from agent_adapter import Registry
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import ApprovalPending
from agent_adapter.core.types import ToolCall
from agent_adapter.exec import Executor
from agent_adapter.exec._approve import pending_store

registry = Registry()


@registry.tool(confirm=True)
async def pay(amount: int) -> str:
    """Pay an amount (needs approval)."""
    return f"paid {amount}"


def fields(exc: ApprovalPending) -> tuple[object, ...]:
    return (type(exc), exc.args, exc.approval_id, exc.tool, exc.run_id, exc.call_id)


async def pending_result():
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    result = await executor.run_call(ToolCall(name="pay", arguments={"amount": 1}, id="c1"))
    assert result.status == "pending" and isinstance(result.exception, ApprovalPending)
    return result, store


@pytest.mark.parametrize("clone", [lambda v: pickle.loads(pickle.dumps(v)), copy.deepcopy])
async def test_a_pending_tool_result_from_a_store_backed_port_round_trips(clone):
    result, store = await pending_result()
    back = clone(result)
    assert back.to_dict() == result.to_dict()
    assert fields(back.exception) == fields(result.exception)
    assert back.exception.arguments is None  # arguments never leave the process
    assert pending_store(result.exception, None) is store  # a flow can still withdraw it
    assert pending_store(back.exception, None) is None  # the store stays in this process
    caused = ApprovalPending("other")
    caused.__cause__ = result.exception
    assert pending_store(caused, None) is store


@pytest.mark.parametrize("clone", [lambda v: pickle.loads(pickle.dumps(v)), copy.deepcopy])
async def test_the_approval_pending_a_deferred_executor_raises_round_trips(clone):
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    with pytest.raises(ApprovalPending) as caught:
        await executor.run("pay", {"amount": 2})
    exc = caught.value
    back = clone(exc)
    assert fields(back) == fields(exc) and exc.arguments == {"amount": 2}
    assert back.arguments is None
    assert not [key for key in back.__dict__ if key.startswith("_aa_")]


def test_private_keys_are_never_pickled():
    exc = ApprovalPending("a1", tool="pay")
    exc.__dict__["_aa_anything"] = object()
    back = pickle.loads(pickle.dumps(exc))
    assert "_aa_anything" not in back.__dict__ and back.approval_id == "a1"
