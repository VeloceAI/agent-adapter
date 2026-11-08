"""Golden path: the support desk as a FastAPI service. Run: uvicorn golden_path.app:app"""

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent_adapter import AdapterError

from .agent import SupportDesk, default_desk

# Demo tokens: verify a session or a JWT from your identity provider instead.
USERS = {
    "ann-token": ("ann", "customer"),
    "ben-token": ("ben", "customer"),
    "lena-token": ("lena", "reviewer"),
}
STATUS = {
    "tool-args-invalid": 422,
    "guardrail-blocked": 422,
    "flow-error": 422,
    "forbidden": 403,
    "approval-not-found": 404,
    "approval-already-decided": 409,
    "budget-exceeded": 429,
}


class Ask(BaseModel):
    chat_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=4000)


class Decision(BaseModel):
    approved: bool


class Escalation(BaseModel):
    order_id: str = Field(min_length=1, max_length=32)
    summary: str = Field(min_length=1, max_length=2000)


def caller(authorization: Annotated[str, Header()] = "") -> tuple[str, str]:
    """The signed-in user and role from the bearer token; 401 for anything else."""
    token = authorization.removeprefix("Bearer ").encode("utf-8")
    for known, who in USERS.items():
        if hmac.compare_digest(token, known.encode("utf-8")):
            return who
    raise HTTPException(401, "unknown token")


Caller = Annotated[tuple[str, str], Depends(caller)]


def current_desk(request: Request) -> SupportDesk:
    """The desk startup built (or was given)."""
    desk: SupportDesk = request.app.state.desk
    return desk


Desk = Annotated[SupportDesk, Depends(current_desk)]


# --8<-- [start:app]
def create_app(desk: SupportDesk | None = None) -> FastAPI:
    """The HTTP routes; startup builds the default desk (unless given), recovers crashed flows."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        support = desk if desk is not None else default_desk()
        app.state.desk = support
        await support.recover()  # in-doubt runs are logged and wait for a person
        yield
        support.close()

    app = FastAPI(title="Support desk", lifespan=lifespan)

    @app.exception_handler(AdapterError)
    async def adapter_error(request: Request, error: AdapterError) -> JSONResponse:
        detail = {"kind": error.kind, "message": str(error)}
        return JSONResponse({"detail": detail}, status_code=STATUS.get(error.kind, 400))

    @app.post("/chat")
    async def chat(body: Ask, who: Caller, support: Desk) -> dict[str, str]:
        """One agent turn; the chat id is the caller's own (prefixed with the user)."""
        user = who[0]
        reply = await support.ask(user, f"{user}:{body.chat_id}", body.text)
        return {"text": reply.text, "run_id": reply.run_id, "approval_id": reply.approval_id}

    @app.post("/approvals/{approval_id}")
    async def decide(
        approval_id: str, body: Decision, who: Caller, support: Desk
    ) -> dict[str, str]:
        """A reviewer approves or denies a parked call; it runs at most once."""
        user, role = who
        if role != "reviewer":
            raise HTTPException(403, "only reviewers decide approvals")
        return {"result": await support.decide(approval_id, body.approved, user)}

    @app.post("/tickets")
    async def escalate(body: Escalation, who: Caller, support: Desk) -> dict[str, Any]:
        """Open a ticket and page the team, as a crash-safe flow."""
        ticket = await support.escalate(who[0], body.order_id, body.summary)
        return ticket.model_dump()

    @app.get("/budget")
    async def budget(who: Caller, support: Desk) -> dict[str, Any]:
        """What the caller has spent today, and what is left."""
        status = await support.budget.status(support.context(who[0], ""))
        return {"spent_usd": status.spent_usd, "remaining_usd": status.remaining_usd}

    return app


# --8<-- [end:app]

app = create_app()  # builds nothing until uvicorn starts it
