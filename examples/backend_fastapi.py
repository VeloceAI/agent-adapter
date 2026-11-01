"""FastAPI: one Executor per process, async routes, approvals over HTTP, the tool server mounted."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from backend_common import (
    Backend,
    ask,
    build_backend,
    decide,
    invoke,
    registry,
    tool_server_options,
    user_for,
)
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_adapter.runtime import LoopLocal
from agent_adapter.server import create_app


class InvokeBody(BaseModel):
    arguments: dict[str, Any]


class DecideBody(BaseModel):
    approved: bool
    edited_arguments: dict[str, Any] | None = None


class AskBody(BaseModel):
    question: str


def current_user(authorization: Annotated[str | None, Header()] = None) -> str:
    """Your auth dependency; here a token lookup."""
    user = user_for(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="unknown token")
    return user


User = Annotated[str, Depends(current_user)]


# --8<-- [start:app]
def make_app(backend: Backend | None = None) -> FastAPI:
    """The app; `backend` defaults to one built here, once per worker process."""
    service = backend if backend is not None else build_backend()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await LoopLocal.aclose_all()  # close loop-bound clients before the server's loop ends

    app = FastAPI(lifespan=lifespan)

    @app.post("/tools/{name}")
    async def call_tool(name: str, body: InvokeBody, user: User) -> JSONResponse:
        status, reply = await invoke(service, name, body.arguments)  # await: same loop
        return JSONResponse(reply, status_code=status)

    @app.post("/approvals/{approval_id}/decide")
    async def decide_call(approval_id: str, body: DecideBody, user: User) -> JSONResponse:
        status, reply = await decide(
            service,
            approval_id,
            reviewer=user,
            approved=body.approved,
            edited_args=body.edited_arguments,
        )
        return JSONResponse(reply, status_code=status)

    @app.post("/ask")
    async def ask_model(body: AskBody, user: User) -> JSONResponse:
        status, reply = await ask(service, body.question)
        return JSONResponse(reply, status_code=status)

    # --8<-- [start:mount]
    tools = create_app(registry, mcp=False, **tool_server_options(service))
    app.mount("/agent", tools)  # REST tool server: /agent/tools, /agent/approvals, ...
    # --8<-- [end:mount]
    return app


# --8<-- [end:app]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(make_app(), host="127.0.0.1", port=8000)  # one worker: one Executor
