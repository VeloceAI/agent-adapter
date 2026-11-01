"""Support for the Claude Agent SDK tests: MCP calls, the bundled CLI, a scripted Messages API."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import claude_agent_sdk
import mcp
import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import build_spec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.integrations import claude_agent
from tests.fakes.approval import RecordingApproval
from tests.integrations.kit import FakeRunner

TOOL = "mcp__agent_adapter__remove"
TOOL_USE_ID_META = "claudecode/toolUseId"  # where Claude Code puts the tool-use id in tools/call
CLI = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude.exe"
needs_cli = pytest.mark.skipif(
    not (CLI.exists() or (CLI.parent / "claude").exists()),
    reason="the claude-agent-sdk wheel bundles no CLI here",
)


def text_of(result: dict[str, Any]) -> Any:
    (block,) = result["content"]
    try:
        return json.loads(block["text"])
    except ValueError:
        return block["text"]


def as_handler_result(result: Any) -> dict[str, Any]:
    """An MCP `CallToolResult` in the handler's own result shape."""
    content = [{"type": "text", "text": block.text} for block in result.content]
    return {"content": content, "is_error": bool(result.is_error)}


async def call_server(
    server: dict[str, Any], name: str, args: dict[str, Any], tool_use_id: str | None = None
) -> dict[str, Any]:
    """One `tools/call` through the SDK MCP server, as Claude Code sends it (with its id)."""
    meta = {TOOL_USE_ID_META: tool_use_id} if tool_use_id is not None else None
    async with mcp.Client(server["instance"]) as client:
        return as_handler_result(await client.call_tool(name, args, meta=meta))


async def dispatch(
    bridge: Any, name: str, args: dict[str, Any], tool_use_id: str
) -> dict[str, Any]:
    """The bridge server's own `tools/call` handler, as its MCP session invokes it."""
    entry = bridge.server["instance"].get_request_handler("tools/call")
    payload = {"name": name, "arguments": args, "_meta": {TOOL_USE_ID_META: tool_use_id}}
    return as_handler_result(await entry.handler(None, entry.params_type.model_validate(payload)))


def sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def turn(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Ask for `MessagesApi.calls` once per prompt, then finish with text ("hello": text)."""
    last = body.get("messages", [])[-1]
    content = last.get("content")
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    answered = any(b.get("type") == "tool_result" for b in blocks) or any(
        "hello" in str(b.get("text", "")) for b in blocks
    )
    start = {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
             "content": [], "stop_reason": None, "stop_sequence": None,
             "usage": {"input_tokens": 1, "output_tokens": 1}}  # fmt: skip
    if answered:
        block, delta, stop = (
            {"type": "text", "text": ""},
            {"type": "text_delta", "text": "ok"},
            "end_turn",
        )
        blocks_out = [(block, delta)]
    else:
        stop = "tool_use"
        blocks_out = [
            (
                {"type": "tool_use", "id": use_id, "name": TOOL, "input": {}},
                {"type": "input_json_delta", "partial_json": json.dumps(arguments)},
            )
            for use_id, arguments in MessagesApi.calls
        ]
    events: list[dict[str, Any]] = [{"type": "message_start", "message": start}]
    for index, (block, delta) in enumerate(blocks_out):
        events += [
            {"type": "content_block_start", "index": index, "content_block": block},
            {"type": "content_block_delta", "index": index, "delta": delta},
            {"type": "content_block_stop", "index": index},
        ]
    return [
        *events,
        {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]


class MessagesApi(BaseHTTPRequestHandler):
    calls: list[tuple[str, dict[str, Any]]] = [("toolu_01", {"item": "a"})]  # noqa: RUF012

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        data = sse(turn(body))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def api() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MessagesApi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


class Reviewer:
    """Approves everything; records what it was asked."""

    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return ApprovalDecision(approved=True)


async def run_cli(
    api: str,
    home: Path,
    bridge: Any,
    *,
    prompts: tuple[str, ...] = ("remove a",),
    hooks: Any = None,
    **options: Any,
) -> list[Any]:
    env = {
        "ANTHROPIC_BASE_URL": api,
        "ANTHROPIC_API_KEY": "sk-ant-offline-test",
        "CLAUDE_CONFIG_DIR": str(home),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_AUTOUPDATER": "1",
        "ENABLE_TOOL_SEARCH": "false",  # list MCP tools directly
    }
    config = ClaudeAgentOptions(
        mcp_servers={bridge.name: bridge.server}, can_use_tool=bridge.can_use_tool,
        hooks=bridge.hooks if hooks is None else hooks, env=env, cwd=str(home),
        setting_sources=[], max_turns=3, model="claude-sonnet-4-5", **options,
    )  # fmt: skip
    messages: list[Any] = []
    async with ClaudeSDKClient(options=config) as client:
        for prompt in prompts:
            await client.query(prompt)
            messages += [message async for message in client.receive_response()]
    return messages


def cli_setup(port: Any, *, executor_port: Any = None) -> tuple[Any, list[str], FakeRunner]:
    ran: list[str] = []

    async def remove(item: str) -> str:
        """Remove an item."""
        ran.append(item)
        return f"removed {item}"

    registry = Registry()
    registry.add(build_spec(remove, confirm=True))
    # By default no port of its own: without a token the call is denied.
    runner = FakeRunner(approval=RecordingApproval(executor_port) if executor_port else None)
    return claude_agent.claude_sdk_bridge(registry, port, executor=runner), ran, runner
