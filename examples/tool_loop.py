"""Tool loop: an instrumented model calls registry tools; the ledger and cache track it all."""

import asyncio

from agent_adapter import (
    CallableModel,
    Decision,
    DeferredApproval,
    Executor,
    Ledger,
    LLMRequest,
    LLMResponse,
    MemoryApprovalStore,
    MemoryCache,
    Message,
    Prices,
    Registry,
    ToolCall,
    Usage,
    run_tool_loop,
)
from agent_adapter.tracing import EventLog

registry = Registry()


@registry.tool
def exchange_rate(currency: str) -> float:
    """Today's EUR rate for a currency."""
    return {"USD": 0.92, "GBP": 1.17}.get(currency, 1.0)


@registry.tool(confirm=True)
def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice in USD (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"


async def scripted_llm(request: LLMRequest) -> LLMResponse:
    """Stands in for a real model: asks for a tool, then answers from its result."""
    last = request.messages[-1]
    usage = Usage(input_tokens=1000, output_tokens=50)
    if last.role == "tool":
        content = f"The {last.name} result is {last.content}."
        return LLMResponse(
            provider="host",
            model="router-v2",
            text=content,
            usage=usage,
            message=Message(role="assistant", content=content),
        )
    if "invoice" in last.content:
        tool, args = "send_invoice", {"customer_id": "cus_1024", "amount": 49.5}
    else:
        tool, args = "exchange_rate", {"currency": "USD"}
    call = ToolCall(name=tool, arguments=args, id=f"call_{tool}")
    return LLMResponse(
        provider="host",
        model="router-v2",
        tool_calls=(call,),
        finish_reason="tool_calls",
        usage=usage,
        message=Message(role="assistant", tool_calls=(call,)),
    )


PRICES = Prices.load(
    {"version": "demo", "models": {"router-v2": {"provider": "host", "input": 2.0, "output": 8.0}}}
)


async def main() -> None:
    # --8<-- [start:loop]
    ledger, sink = Ledger(prices=PRICES), EventLog()
    executor = Executor(registry, ledger=ledger, cache=MemoryCache(), sink=sink)
    model = CallableModel(scripted_llm, provider="host", model="router-v2")
    question = (Message(role="user", content="What is the USD rate?"),)

    first = await run_tool_loop(model, registry, question, executor=executor)
    print(first.finish, first.steps, first.final.text)
    print(f"{ledger.total_usd():.4f}")

    again = await run_tool_loop(model, registry, question, executor=executor)
    print(again.final.text == first.final.text, f"{ledger.total_usd():.4f}")  # cache hits: free
    # --8<-- [end:loop]

    # --8<-- [start:instrumented]
    llm = executor.model(model, cache_ttl=3600)  # hooks, response cache, ledger, events
    async with executor.run_scope() as ctx:
        reply = await llm.bind(ctx).generate(LLMRequest(messages=question, stage="answer"))
    [entry] = ledger.entries(ctx.run_id)
    print(reply.tool_calls[0].name, entry.entry_id.split(":", 1)[1], entry.cache_hit)
    # --8<-- [end:instrumented]

    # --8<-- [start:pending]
    store = MemoryApprovalStore()
    hitl = executor.replace(approval=DeferredApproval(store), cache=None)
    ctx = hitl.context(run_id="run_invoice")
    ask = (Message(role="user", content="Please invoice cus_1024 for 49.50 USD."),)
    paused = await run_tool_loop(model, registry, ask, executor=hitl, ctx=ctx)
    print(paused.finish)

    [waiting] = await store.list(run_id="run_invoice", status="pending")
    await store.decide(waiting.approval_id, Decision(approved=True, decided_by="ann"))
    done = await run_tool_loop(model, registry, paused.messages, executor=hitl, ctx=ctx)
    print(done.finish, done.final.text)
    # --8<-- [end:pending]
    print(len(sink.of_type("llm.cached")), len(sink.of_type("llm.finished")))


if __name__ == "__main__":
    asyncio.run(main())
