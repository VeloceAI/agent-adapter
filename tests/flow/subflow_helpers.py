"""State models and flows shared by the sub-flow tests."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel

from agent_adapter.flow import Call, Flow, Merge, SubFlow


class Order(BaseModel):
    item: str = ""
    qty: int = 0
    note: str = ""
    log: Annotated[list[str], Merge.append] = []
    total: Annotated[int, Merge.sum] = 0


class Pack(BaseModel):
    item: str = ""
    boxes: int = 0
    label: str = ""


class Shop(BaseModel):
    item: str = ""
    qty: int = 0
    packed: Pack | None = None
    label: str = ""
    log: Annotated[list[str], Merge.append] = []


def count(state: Order) -> dict:
    return {"qty": state.qty + 1, "log": [f"count {state.item}"], "total": 1}


def child_flow(name: str = "child", *, gate: bool = False) -> Flow[Order]:
    """`count` then `check` (gated when `gate`)."""
    return (
        Flow(name, state=Order)
        .node("count", count)
        .node("check", lambda s: {"note": f"checked {s.qty}", "log": ["check"]}, pause=gate)
        .edge("count", "check")
        .end("check")
    )


def pay_flow() -> Flow[Order]:
    """A child whose second node calls the `pay` tool (needs approval)."""
    return (
        Flow("payer", state=Order)
        .node("count", count)
        .node("pay", Call("pay", args=lambda s: {"amount": s.qty}, into="note"))
        .edge("count", "pay")
        .end("pay")
    )


def parent_flow(child: Flow[Order], name: str = "parent", **options) -> Flow[Order]:
    """`start` -> SubFlow `sub` -> `after`."""
    return (
        Flow(name, state=Order)
        .node("start", lambda s: {"qty": 2, "log": ["start"]})
        .node("sub", SubFlow(child, **options))
        .node("after", lambda s: {"log": [f"after {s.note}"]})
        .edge("start", "sub")
        .edge("sub", "after")
        .end("after")
    )
