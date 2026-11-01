"""The service layer the backend examples share: tools, one Executor per process, approvals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agent_adapter import (
    AdapterError,
    ApprovalDecision,
    ApprovalPending,
    BudgetHook,
    CallableModel,
    DeferredApproval,
    Executor,
    Ledger,
    LLMRequest,
    LoggingHook,
    MemoryApprovalStore,
    Message,
    Prices,
    Registry,
    RunLimits,
    ToolCall,
    Usage,
    run_tool_loop,
)
from agent_adapter.approval import ApprovalStore
from agent_adapter.errors import ApprovalAlreadyDecided, Forbidden
from agent_adapter.runtime import new_id
from agent_adapter.server import APPROVER_SCOPE, Principal, bearer_auth, problem_for

logger = logging.getLogger(__name__)

Reply = tuple[int, dict[str, Any]]  # (HTTP status, JSON body)

registry = Registry()


# --8<-- [start:tools]
@registry.tool(effects="read_only", idempotent=True, timeout=5, retries=2)
async def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@registry.tool(confirm=True, effects="destructive", timeout=10)
def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice to a customer (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"  # a sync tool runs in a worker thread


# --8<-- [end:tools]

# A stand-in for your auth: bearer token -> user. The reviewer may decide approvals.
TOKENS = {"dev-token": "dev@example.com", "reviewer-token": "reviewer@example.com"}
REVIEWERS = frozenset({"reviewer@example.com"})

PRICES = Prices.load(
    {"version": "demo", "models": {"router-v2": {"provider": "host", "input": 2.0, "output": 8.0}}}
)


async def scripted_llm(request: LLMRequest) -> dict[str, Any]:
    """Stands in for a real model: asks for a shipping rate, then answers from the result."""
    last = request.messages[-1]
    usage = Usage(input_tokens=1000, output_tokens=50)  # $0.0024 per call at PRICES
    if last.role == "tool":
        return {"text": f"The shipping rate is {last.content}%.", "usage": usage}
    call = ToolCall(name="shipping_rate", arguments={"region": "us"}, id="call_rate")
    return {"tool_calls": (call,), "finish_reason": "tool_calls", "usage": usage}


MODEL = CallableModel(scripted_llm, provider="host", model="router-v2")


# --8<-- [start:build]
@dataclass(frozen=True)
class Backend:
    """What one process shares across requests: the executor, its approval store, the ledger."""

    executor: Executor
    store: ApprovalStore
    ledger: Ledger


def build_backend(*, store: ApprovalStore | None = None, max_usd: float = 0.01) -> Backend:
    """Build once per process; pass a database `ApprovalStore` when workers share approvals."""
    store = store if store is not None else MemoryApprovalStore()
    ledger = Ledger(prices=PRICES)
    executor = Executor(
        registry,
        approval=DeferredApproval(store),  # confirm tools park in the store
        ledger=ledger,
        hooks=[LoggingHook(), BudgetHook(max_usd)],  # the budget is per run: one per request
        limits=RunLimits(max_calls=20),
    )
    return Backend(executor, store, ledger)


# --8<-- [end:build]


def user_for(authorization: str | None) -> str | None:
    """The user a `Bearer <token>` header names, else None."""
    token = (authorization or "").removeprefix("Bearer ")
    return TOKENS.get(token)


def tool_server_options(backend: Backend) -> dict[str, Any]:
    """Options for `create_app` / `create_wsgi_app`: this process's executor, its own store."""
    principals = {
        token: Principal(user, scopes=frozenset({APPROVER_SCOPE} if user in REVIEWERS else ()))
        for token, user in TOKENS.items()
    }
    return {
        "executor": backend.executor,  # same hooks, budget and ledger as your own routes
        "approvals": MemoryApprovalStore(),  # decided over the server's own /approvals routes
        "authenticate": bearer_auth(principals),
    }


def error_reply(exc: AdapterError) -> Reply:
    """The RFC 9457 problem the HTTP tool server would send for `exc`."""
    found = problem_for(exc)
    logger.warning("request refused: %s (%d)", found.type, found.status)
    return found.status, found.model_dump()


# --8<-- [start:service]
async def call(
    backend: Backend, tool: str, arguments: dict[str, Any], *, run_id: str, call_id: str
) -> Any:
    """Run one tool call as the run `run_id`; errors raise (`ApprovalPending` included)."""
    async with backend.executor.run_scope(run_id=run_id) as ctx:
        return await backend.executor.run(tool, arguments, ctx, call_id=call_id)


async def invoke(
    backend: Backend,
    tool: str,
    arguments: dict[str, Any],
    *,
    run_id: str | None = None,
    call_id: str | None = None,
) -> Reply:
    """200 with the result, 202 when the call waits for a reviewer, else a problem."""
    run_id, call_id = run_id or new_id("run_"), call_id or new_id("call_")
    try:
        value = await call(backend, tool, arguments, run_id=run_id, call_id=call_id)
    except ApprovalPending as pending:
        return 202, {"status": "pending_approval", "approval_id": pending.approval_id}
    except AdapterError as exc:
        return error_reply(exc)
    return 200, {"status": "ok", "run_id": run_id, "result": value}


async def record_decision(
    backend: Backend,
    approval_id: str,
    *,
    reviewer: str,
    approved: bool,
    edited_args: dict[str, Any] | None = None,
) -> Reply:
    """Check the reviewer, validate edits, store the decision; nothing runs yet."""
    try:
        if reviewer not in REVIEWERS:
            raise Forbidden("only reviewers decide approvals")
        record = await backend.store.get(approval_id)  # ApprovalNotFound -> 404
        if edited_args is not None:  # validate edits before the decision is stored
            registry.get(record.request.tool).validate_args(edited_args)
        decision = ApprovalDecision(approved=approved, edited_args=edited_args, decided_by=reviewer)
        await backend.store.decide(approval_id, decision)  # a second decision -> 409
    except AdapterError as exc:
        return error_reply(exc)
    return 200, {"status": "approved" if approved else "denied", "approval_id": approval_id}


async def decide(
    backend: Backend,
    approval_id: str,
    *,
    reviewer: str,
    approved: bool,
    edited_args: dict[str, Any] | None = None,
) -> Reply:
    """Record a reviewer's decision; an approved call then runs once, in this request."""
    reply = await record_decision(
        backend, approval_id, reviewer=reviewer, approved=approved, edited_args=edited_args
    )
    if reply[0] != 200 or not approved:
        return reply
    return await resume(backend, approval_id)


async def resume(backend: Backend, approval_id: str) -> Reply:
    """Re-run a decided call with its own run and call ids: the store answers it, once."""
    try:
        record = await backend.store.get(approval_id)
        if record.consumed:
            raise ApprovalAlreadyDecided(approval_id, status=record.status)
    except AdapterError as exc:
        return error_reply(exc)
    request = record.request  # the original arguments: the decision is bound to them
    return await invoke(
        backend, request.tool, request.arguments, run_id=request.run_id, call_id=request.call_id
    )


# --8<-- [end:service]


# --8<-- [start:ask]
async def ask(backend: Backend, question: str) -> Reply:
    """Answer with the tool loop; the run's budget caps it and the reply carries its cost."""
    async with backend.executor.run_scope() as ctx:
        messages = (Message(role="user", content=question),)
        try:
            result = await run_tool_loop(
                MODEL, registry, messages, executor=backend.executor, ctx=ctx
            )
        except AdapterError as exc:  # BudgetExceeded -> 429
            return error_reply(exc)
        cost = backend.ledger.total_usd(ctx.run_id)
    return 200, {"status": result.finish, "answer": result.final.text, "cost_usd": round(cost, 6)}


# --8<-- [end:ask]
