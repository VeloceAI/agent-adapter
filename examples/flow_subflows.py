"""Sub-flows and a node cache: a packing flow runs inside an order flow and pauses for a check."""

import asyncio
from typing import Annotated

from pydantic import BaseModel

from agent_adapter import (
    ApprovalDecision,
    DeferredApproval,
    Executor,
    Flow,
    FlowRunner,
    MemoryApprovalStore,
    MemoryCache,
)
from agent_adapter.errors import FlowError
from agent_adapter.flow import Merge, NodeCache, SubFlow

QUOTED: list[str] = []


# --8<-- [start:state]
class Order(BaseModel):
    item: str
    qty: int = 1
    price: int = 0
    label: str = ""
    log: Annotated[list[str], Merge.append] = []


class Parcel(BaseModel):
    item: str = ""
    boxes: int = 0
    label: str = ""


# --8<-- [end:state]


# --8<-- [start:child]
def measure(parcel: Parcel) -> dict[str, int]:
    return {"boxes": max(1, parcel.boxes // 2)}


def write_label(parcel: Parcel) -> dict[str, str]:
    return {"label": f"{parcel.boxes} box(es) of {parcel.item}"}


packing = (
    Flow("packing", state=Parcel)
    .node("measure", measure)
    .node("label", write_label, pause=lambda p: {"boxes": p.boxes})  # a person checks the boxes
    .edge("measure", "label")
    .end("label")
)
# --8<-- [end:child]


# --8<-- [start:parent]
def quote(order: Order) -> dict[str, object]:
    QUOTED.append(order.item)  # stands in for a slow price lookup
    return {"price": 4 * order.qty, "log": [f"quoted {order.item}"]}


order_flow = (
    Flow("order", state=Order)
    .node("quote", quote, cache=NodeCache(reads=["item", "qty"], ttl=600, scope="global"))
    .node("pack", SubFlow(packing, input={"item": "item", "boxes": "qty"}, into={"label": "label"}))
    .node("ship", lambda o: {"log": [f"shipped {o.label} for {o.price}"]})
    .edge("quote", "pack")
    .edge("pack", "ship")
    .end("ship")
)
# --8<-- [end:parent]


# --8<-- [start:run]
async def main() -> None:
    executor = Executor(approval=DeferredApproval(MemoryApprovalStore()), cache=MemoryCache())
    runner = FlowRunner(executor)
    for _ in range(2):  # the second order hits the quote cache
        paused = await runner.run(order_flow, {"item": "mug", "qty": 4})
        if paused.pending is None or paused.checkpoint is None:
            raise FlowError(f"expected the packing check, got {paused.status}")
        print("paused", paused.pending.tool, paused.pending.arguments)
        done = await runner.resume(order_flow, paused.checkpoint, ApprovalDecision(approved=True))
        print(done.status, done.state.log)
    print("quote ran", len(QUOTED), "time(s)")


# --8<-- [end:run]

if __name__ == "__main__":
    asyncio.run(main())
