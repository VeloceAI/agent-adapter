"""AG-UI over HTTP: one `POST /agui` route on Starlette or FastAPI, behind bearer auth."""

import hmac
from typing import Any

from agent_adapter import Executor, Registry
from agent_adapter.integrations.ag_ui import AGUIBridge, AGUIRun, agui_endpoint

registry = Registry()


@registry.tool
def order_total(order_id: str) -> float:
    """The total of an order."""
    return 120.0


async def agent(run: AGUIRun) -> Any:
    total = await run.call("order_total", {"order_id": "o-7"})
    run.text(f"the total is {total.value}")
    return total.value


# --8<-- [start:auth]
TOKENS = {"s3cret-token": "lead-7"}  # load from your secret store, never from source


async def authenticate(request: Any) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    for known, principal in TOKENS.items():
        if scheme.lower() == "bearer" and hmac.compare_digest(token, known):
            return principal
    return None  # 401


# --8<-- [end:auth]

# --8<-- [start:app]
endpoint = agui_endpoint(AGUIBridge(Executor(registry)), agent, authenticate=authenticate)


def starlette_app() -> Any:
    from starlette.applications import Starlette
    from starlette.routing import Route

    return Starlette(routes=[Route("/agui", endpoint, methods=["POST"])])


def fastapi_app() -> Any:
    from fastapi import FastAPI

    app = FastAPI()
    app.add_route("/agui", endpoint, methods=["POST"])
    return app


# --8<-- [end:app]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(starlette_app(), host="127.0.0.1", port=8000)
