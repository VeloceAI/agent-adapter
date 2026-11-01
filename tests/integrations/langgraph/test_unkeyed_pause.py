"""A one_use node cannot pause after a call it cannot replay: it fails closed, never re-runs it."""

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.graph import MessagesState

from agent_adapter.core.errors import ApprovalAlreadyDecided, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.langgraph import LangGraphApproval, one_use
from agent_adapter.integrations.langgraph._spent import CLAIMS
from agent_adapter.integrations.langgraph._user_node import USER_NODE_MEMO
from tests.integrations.langgraph.support import config, one_node, resume

pytestmark = pytest.mark.contract

RAN: Counter[str] = Counter()
APPROVE = ApprovalDecision(approved=True)
registry = Registry()


@registry.tool(confirm=True, portable=False)
async def odd(target: Path) -> str:
    """Touch a path (needs approval; a Path argument has no journal key)."""
    RAN["odd"] += 1
    return str(target)


@registry.tool(confirm=True)
async def wire(text: str) -> str:
    """Wire money (needs approval)."""
    RAN[f"wire {text}"] += 1
    return f"wired {text}"


@pytest.fixture(autouse=True)
def _clean():
    RAN.clear()
    CLAIMS.clear()
    USER_NODE_MEMO.clear()


async def test_a_pause_after_an_unkeyed_confirm_call_fails_closed():
    executor = Executor(registry, approval=LangGraphApproval())

    async def node(state: Any) -> dict[str, Any]:
        ctx = executor.context(run_id="user-run")
        await executor.run("odd", {"target": Path("x")}, ctx, call_id="o1")
        await executor.run("wire", {"text": "b"}, ctx, call_id="w1")
        return {"messages": []}

    app, cfg = one_node(one_use(node), MessagesState), config()
    [first] = (await app.ainvoke({"messages": []}, cfg, version="v2")).interrupts
    with pytest.raises(ToolUsageError, match="cannot pause after a call it cannot replay"):
        await app.ainvoke(resume(first, APPROVE), cfg)  # odd ran, then wire b would pause
    with pytest.raises(ApprovalAlreadyDecided):  # the host retries the failed task
        await app.ainvoke(None, cfg)
    assert RAN == {"odd": 1}
    assert len(USER_NODE_MEMO) == 0
