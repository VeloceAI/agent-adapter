"""Golden path: the support desk agent: one executor with approvals, budgets, guards and a recorder.

A scripted model stands in for a real one, so it runs with no network and no API key.
"""

import atexit
import functools
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_adapter import (
    AdapterError,
    BudgetHook,
    CallableModel,
    DeferredApproval,
    Executor,
    Flow,
    FlowRunner,
    Ledger,
    LLMRequest,
    LLMResponse,
    Message,
    Prices,
    ToolCall,
    ToolContext,
    Usage,
    run_tool_loop,
)
from agent_adapter.chat import ChatMessage, ChatSession
from agent_adapter.errors import ApprovalNotFound, Forbidden
from agent_adapter.exec.checkpoints import SQLiteCheckpointStore
from agent_adapter.flow import Call
from agent_adapter.hooks import PIIGuard
from agent_adapter.hooks.builtin.budgets import SQLiteBudgetStore
from agent_adapter.memory import SQLiteMemory
from agent_adapter.viewer import DEFAULT_FOLDER, record

from .requesters import Requesters
from .tools import CHAT_TOOLS, build_registry

logger = logging.getLogger(__name__)
SYSTEM = "You are the shop's support desk. Use the tools; cite help pages; never guess."
# Illustrative prices (USD per 1M tokens); load your own table in real code.
PRICES = Prices.load(
    {
        "version": "demo",
        "models": {"desk-v1": {"provider": "local", "input": 1.0, "output": 4.0}},
        "units": {"local:words-v1": {"provider": "local", "per_unit": 0.0, "unit": "1M tokens"}},
    }
)
USAGE = Usage(input_tokens=400, output_tokens=40)  # what each scripted reply "costs"


# --8<-- [start:model]
def scripted_model(request: LLMRequest) -> LLMResponse:
    """Stands in for a real model: picks one tool for the question, then answers from its result."""
    last = request.messages[-1]
    if last.role == "tool":
        return _reply(Message(role="assistant", content=_answer(last.name, last.content)))
    call = ToolCall(id="call_1", **_pick_tool(last.content))
    return _reply(Message(role="assistant", tool_calls=(call,)))


# --8<-- [end:model]


def _pick_tool(text: str) -> dict[str, Any]:
    order = re.search(r"\b\d{4}\b", text)
    lowered = text.lower()
    if lowered.startswith("remember "):
        return {"name": "remember", "arguments": {"text": text[len("remember ") :]}}
    if "about me" in lowered:
        return {"name": "recall", "arguments": {"query": ""}}
    if order and "refund" in lowered:
        return {"name": "refund_order", "arguments": {"order_id": order.group()}}
    if order:
        return {"name": "order_status", "arguments": {"order_id": order.group()}}
    return {"name": "search_help", "arguments": {"query": text}}


def _answer(tool: str, content: str) -> str:
    if content.startswith('{"error"'):
        return f"I could not do that ({json.loads(content)['error']['kind']})."
    if tool == "order_status":
        return f"Your order is {content}."
    if tool == "search_help":
        best = json.loads(content)["documents"][0]
        return f"{best['text']} [source: {best['source']['uri']}]"
    if tool == "recall":
        facts = [item["text"] for item in json.loads(content)]
        return f"From earlier chats: {'; '.join(facts)}" if facts else "Nothing saved about you."
    return "I will remember that." if tool == "remember" else content


def _reply(message: Message) -> LLMResponse:
    return LLMResponse(
        provider="local",
        model="desk-v1",
        text=message.content,
        tool_calls=message.tool_calls,
        finish_reason="tool_calls" if message.tool_calls else "stop",
        message=message,
        usage=USAGE,
    )


# --8<-- [start:flow]
class Ticket(BaseModel):
    order_id: str
    summary: str
    ticket_id: str = ""
    note: str = ""


def ticket_args(ticket: Ticket) -> dict[str, str]:
    """The arguments of `open_ticket`."""
    return {"order_id": ticket.order_id, "summary": ticket.summary}


ESCALATION = (
    Flow("escalation", state=Ticket)
    .node("open", Call("open_ticket", args=ticket_args, into="ticket_id"))
    .node("notify", Call("notify_team", args=lambda t: {"ticket_id": t.ticket_id}, into="note"))
    .edge("open", "notify")
    .end("notify")
)
# --8<-- [end:flow]


@dataclass(frozen=True)
class Reply:
    """One answer; `approval_id` is set when a confirm tool waits for a reviewer."""

    text: str
    run_id: str
    approval_id: str = ""


# --8<-- [start:desk]
class SupportDesk:
    """Everything one process needs, built once: tools, stores, the executor and the flow runner."""

    def __init__(self, folder: Path, *, runs: Path, daily_usd: float = 0.50) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.memory = SQLiteMemory(folder / "memory.db", retention_days=90)
        self.registry = build_registry(self.memory)
        self.chat_tools = self.registry.subset(names=CHAT_TOOLS)
        self.budget = BudgetHook(
            daily_usd, scope="user", per="day", store=SQLiteBudgetStore(folder / "budgets.db")
        )
        self.executor = Executor(
            self.registry,
            hooks=[
                # Block cards and secrets in what the user sends the model and in tool arguments.
                PIIGuard(kinds=["secret", "card"], on=("llm_request", "args")),
                PIIGuard(on=("result", "llm_response"), action="redact"),  # mask on the way out
                self.budget,  # per user per UTC day, shared by every worker on this file
            ],
            approval=DeferredApproval.sqlite(folder / "approvals.db"),  # survives a restart
            ledger=Ledger(prices=PRICES),
        )
        self.requesters = Requesters(folder / "requesters.db")
        self.recorder = record(self.executor, folder=runs)  # agent-adapter ui
        store = SQLiteCheckpointStore(folder / "checkpoints.db")
        self.runner = FlowRunner(
            self.executor, checkpoints=store, checkpoint_key=checkpoint_key(folder)
        )
        self.model = CallableModel(scripted_model, provider="local", model="desk-v1")

    def context(self, user: str, chat_id: str, run_id: str | None = None) -> ToolContext:
        """A call context; the host sets the user, never the model."""
        return self.executor.context(run_id=run_id, thread_id=chat_id, user_id=user)

    async def ask(self, user: str, chat_id: str, text: str, run_id: str | None = None) -> Reply:
        """One turn of the agent loop for `user` in chat `chat_id`.

        Raises:
            BudgetExceeded: the user's daily budget is spent.
        """
        ctx = self.context(user, chat_id, run_id)
        messages = [Message(role="system", content=SYSTEM), Message(role="user", content=text)]
        result = await run_tool_loop(
            self.model, self.chat_tools, messages, executor=self.executor, ctx=ctx
        )
        if result.finish == "pending":
            waiting = next(r for r in result.tool_results if r.status == "pending")
            approval_id = str(waiting.error.details["approval_id"]) if waiting.error else ""
            text = f"A reviewer must approve {waiting.name} first."
            await self.requesters.add(approval_id, user, ctx.run_id)
            return Reply(text=text, run_id=ctx.run_id, approval_id=approval_id)
        return Reply(text=result.final.text or "", run_id=ctx.run_id)

    async def decide(self, approval_id: str, approved: bool, reviewer: str) -> str:
        """Apply a reviewer's decision; an approved call runs once, here, as its requester.

        Raises:
            ApprovalNotFound: unknown id.
            Forbidden: the reviewer asked for the call (four eyes).
            ApprovalAlreadyDecided: the decision was used already.
        """
        asked = await self.requesters.get(approval_id)
        if asked is None:
            raise ApprovalNotFound(approval_id)
        requester, run_id = asked
        if reviewer == requester:
            raise Forbidden("a reviewer cannot decide their own request")
        ctx = self.executor.context(run_id=run_id, user_id=requester)  # tools check ownership
        try:
            result = await self.executor.resume(approval_id, approved, ctx, decided_by=reviewer)
        finally:
            self.executor.end_run(run_id)
        why = result.error.kind if result.error else result.status
        return str(result.value) if result.ok else f"not run ({why})"

    async def escalate(self, user: str, order_id: str, summary: str) -> Ticket:
        """Run the escalation flow, checkpointed after every step."""
        ctx = self.executor.context(user_id=user)
        result = await self.runner.run(ESCALATION, {"order_id": order_id, "summary": summary}, ctx)
        return result.state

    async def recover(self, *, rerun_in_doubt: bool = False) -> dict[str, str]:
        """Finish escalations a crashed worker left behind (call on startup)."""
        found = await self.runner.recover_all(ESCALATION, rerun_in_doubt=rerun_in_doubt)
        outcome: dict[str, str] = {}
        for run_id, result in found.items():
            if isinstance(result, AdapterError):
                logger.warning("escalation %s not recovered: %s", run_id, result.kind)
                outcome[run_id] = result.kind
            else:
                outcome[run_id] = result.status
        return outcome

    def close(self) -> None:
        """Flush and detach the run recorder."""
        self.recorder.stop()


# --8<-- [end:desk]


def checkpoint_key(folder: Path) -> bytes:
    """The checkpoint signing key: SUPPORT_DESK_KEY, else a key file made once in `folder`."""
    given = os.environ.get("SUPPORT_DESK_KEY", "")
    if given:
        return given.encode("utf-8")
    path = folder / "checkpoint.key"
    try:
        with path.open("xb") as file:  # exclusive: two workers starting together agree
            file.write(secrets.token_bytes(32))
    except FileExistsError:
        pass  # made by an earlier start or another worker: use it
    return path.read_bytes()


# --8<-- [start:entry]
def build_desk(
    data_dir: Path | None = None, *, runs_dir: Path | None = None, daily_usd: float = 0.50
) -> SupportDesk:
    """A desk on `data_dir` ($SUPPORT_DESK_DATA, else .support_desk); runs go to `runs_dir`.

    The runs default to the viewer's folder, the one `agent-adapter ui` and `eval` use.
    """
    data = data_dir or Path(os.environ.get("SUPPORT_DESK_DATA", ".support_desk"))
    return SupportDesk(data, runs=runs_dir or Path(DEFAULT_FOLDER), daily_usd=daily_usd)


@functools.cache
def default_desk() -> SupportDesk:
    """The process's desk, built on first use (never at import)."""
    return build_desk()


async def chat(message: ChatMessage, session: ChatSession) -> str:
    """The chat-page agent: one turn on the default desk; an anonymous chat is its own user."""
    user = session.user_id or f"guest-{session.id}"  # never one shared memory or budget
    reply = await default_desk().ask(user, session.id, message.text, session.run_id)
    return reply.text


# --8<-- [end:entry]


# --8<-- [start:eval]
EVAL_USER = "ann"  # the demo customer whose orders the eval cases ask about


@functools.cache
def eval_desk() -> SupportDesk:
    """A desk on fresh temporary stores: an eval never parks a live refund or spends a budget."""
    folder = Path(tempfile.mkdtemp(prefix="support-desk-eval-"))
    desk = build_desk(folder)  # runs still go to the viewer's folder, to inspect failures
    atexit.register(_discard, desk, folder)
    return desk


async def eval_chat(message: ChatMessage, session: ChatSession) -> str:
    """The eval agent: the chat agent's turn, on the eval desk, as the demo customer."""
    reply = await eval_desk().ask(EVAL_USER, session.id, message.text, session.run_id)
    return reply.text


def __getattr__(name: str) -> Executor:
    """`eval_executor`, built on first access (for `agent-adapter eval --executor`)."""
    if name == "eval_executor":
        return eval_desk().executor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --8<-- [end:eval]


def _discard(desk: SupportDesk, folder: Path) -> None:
    """Close the eval desk and delete its temporary stores at exit."""
    desk.close()
    if not folder.exists():
        return
    try:
        shutil.rmtree(folder)
    except OSError as exc:  # a store still open elsewhere: leave the temp folder behind
        logger.warning("eval stores not deleted: %s", type(exc).__name__)
