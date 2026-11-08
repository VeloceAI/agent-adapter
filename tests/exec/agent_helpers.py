"""Scripted sub-agent models and a small shop registry for the agent-tool tests."""

from __future__ import annotations

from typing import Any

from agent_adapter import CallableModel, LLMRequest, LLMResponse, Message, Registry, ToolCall, Usage


def reply(text: str = "", calls: tuple[ToolCall, ...] = ()) -> LLMResponse:
    return LLMResponse(
        provider="host",
        model="scripted",
        text=text,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
        usage=Usage(input_tokens=10, output_tokens=2),
        message=Message(role="assistant", content=text, tool_calls=calls),
    )


class Scripted(CallableModel):
    """Calls `tool` with `args` for a user turn, answers with the tool result after it."""

    def __init__(
        self, tool: str = "", args: dict[str, Any] | None = None, call_id: str = ""
    ) -> None:
        self.requests: list[LLMRequest] = []
        self.tool, self.args, self.call_id = tool, args or {}, call_id or f"c_{tool}"
        super().__init__(self.answer, provider="host", model="scripted")

    def answer(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        last = request.messages[-1]
        if last.role == "tool":
            return reply(f"done: {last.content}")
        if not self.tool:
            return reply(f"answer to {last.content}")
        return reply(calls=(ToolCall(name=self.tool, arguments=self.args, id=self.call_id),))


def shop() -> Registry:
    registry = Registry()

    @registry.tool
    def stock(item: str) -> int:
        """Units in stock."""
        return {"lamp": 3}.get(item, 0)

    @registry.tool(confirm=True)
    def refund(order: str) -> str:
        """Refund an order (needs approval)."""
        return f"refunded {order}"

    return registry
