"""Starter FastAPI app: POST /tools/{name} runs one tool. Run: uvicorn app:app --reload"""

from typing import Any

from fastapi import FastAPI, HTTPException

from agent_adapter import AdapterError, Executor, Registry

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


executor = Executor(registry)  # one per process: add hooks=, approval=, limits= here
STATUS = {"tool-not-found": 404, "tool-args-invalid": 422, "approval-denied": 403}
app = FastAPI()


@app.post("/tools/{name}")
async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The JSON body is the tool's arguments; errors become 4xx with the error kind."""
    try:
        return {"result": await executor.run(name, arguments)}
    except AdapterError as error:
        detail = {"kind": error.kind, "message": str(error)}
        raise HTTPException(STATUS.get(error.kind, 400), detail) from error
