"""A model call made inside a flow node records the node in its ledger entry."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agent_adapter.core.context import NODE_KEY, ToolContext
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.registry import Registry
from agent_adapter.cost.ledger import Ledger
from agent_adapter.exec.executor import Executor
from agent_adapter.flow import Flow, FlowRunner, node_model
from agent_adapter.models.callable import CallableModel


class State(BaseModel):
    text: str = ""


def ask() -> LLMRequest:
    return LLMRequest(messages=(Message(role="user", content="hi"),), stage="draft")


async def write(state: State, ctx: ToolContext) -> dict[str, Any]:
    model = node_model(ctx)
    assert model is not None and ctx.extras[NODE_KEY] == "write"
    return {"text": (await model.generate(ask())).text}


async def test_the_ledger_entry_names_the_node_and_a_plain_call_names_none():
    ledger = Ledger()
    executor = Executor(Registry(), ledger=ledger)
    model = CallableModel(lambda r: "ok", provider="p", model="m")
    flow = Flow("notes", state=State).node("write", write).end("write")
    result = await FlowRunner(executor, model=model).run(flow, {})
    assert result.state.text == "ok"
    (entry,) = [e for e in ledger.entries() if e.kind == "llm"]
    assert entry.metadata["node"] == "write" and entry.stage == "draft"

    await executor.model(model).bind(executor.context(run_id="plain")).generate(ask())
    (plain,) = ledger.entries("plain")
    assert "node" not in plain.metadata
