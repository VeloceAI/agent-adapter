"""Shared by the model recipes: vendor SDK errors as typed model errors, and one priced demo run."""

from typing import Any

from agent_adapter import CallableModel, Executor, Ledger, Message, Prices, Registry, run_tool_loop
from agent_adapter.errors import ModelError, ModelTimeout, ModelUnavailable
from agent_adapter.models.errors import class_for_status, model_error, with_overflow


# --8<-- [start:errors]
def typed_error(exc: Exception, *, provider: str, model: str) -> ModelError | None:
    """The SDK error as a `ModelError`, or None for one that is not a provider error."""
    # Typed errors let retries, budgets and FallbackModel tell a rate limit from a bug.
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and isinstance(response, dict):  # botocore's ClientError
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if isinstance(status, int):
        cls = with_overflow(class_for_status(status), exc, getattr(exc, "body", None))
        return model_error(cls, exc, provider=provider, model=model, status_code=status)
    name = type(exc).__name__  # SDKs name their transport errors alike
    if "Timeout" in name:
        return model_error(ModelTimeout, exc, provider=provider, model=model)
    if "Connection" in name:
        return model_error(ModelUnavailable, exc, provider=provider, model=model)
    return None


# --8<-- [end:errors]

registry = Registry()


@registry.tool
def ticket_status(ticket: str) -> str:
    """The status of a support-desk ticket."""
    return {"T-1": "waiting for parts"}.get(ticket, "unknown")


async def demo(model: CallableModel, prices: Prices) -> None:
    """Run the tool loop on `model`, then print the answer and the priced ledger."""
    ledger = Ledger(prices=prices)
    executor = Executor(registry, ledger=ledger)
    ask = (Message(role="user", content="Where is ticket T-1?"),)
    result = await run_tool_loop(model, registry, ask, executor=executor)
    print(model.provider, result.finish, result.steps, result.final.text)
    costs = [entry.cost_usd for entry in ledger.entries() if entry.kind == "llm"]
    print(model.model, [f"{cost:.6f}" for cost in costs if cost is not None])


def tool_turn(messages: list[Any]) -> bool:
    """Fake-SDK helper: True when the last message is a tool result (time to answer)."""
    last = messages[-1]
    role = last.get("role") if isinstance(last, dict) else None
    return role == "tool" or (role == "user" and "toolResult" in str(last))
