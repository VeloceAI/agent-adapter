"""Starter Flask app: POST /tools/<name> runs one tool. Run: flask --app app run"""

from typing import Any

from flask import Flask, jsonify, request

from agent_adapter import AdapterError, Executor, Registry, run_sync

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


executor = Executor(registry)  # one per process: add hooks=, approval=, limits= here
STATUS = {"tool-not-found": 404, "tool-args-invalid": 422, "approval-denied": 403}
app = Flask(__name__)


@app.post("/tools/<name>")
def call_tool(name: str) -> Any:
    """The JSON body is the tool's arguments; errors become 4xx with the error kind."""
    arguments = request.get_json(silent=True) or {}
    try:
        return jsonify(result=run_sync(executor.run(name, arguments), timeout=30))
    except AdapterError as error:
        return jsonify(kind=error.kind, message=str(error)), STATUS.get(error.kind, 400)
