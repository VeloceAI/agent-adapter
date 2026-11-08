"""Amazon Bedrock's Converse API (boto3) wrapped as a CallableModel, with usage and prices."""

import asyncio
from typing import Any

from models_common import demo, tool_turn, typed_error

from agent_adapter import CallableModel, LLMRequest, Message, Prices, ToolCall, Usage

# --8<-- [start:convert]
STOP = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls",
        "max_tokens": "length", "guardrail_intervened": "content_filter",
        "content_filtered": "content_filter"}  # fmt: skip


def _turn(message: Message) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"text": message.content}] if message.content else []
    content += [
        {"toolUse": {"toolUseId": call.id, "name": call.name, "input": call.arguments}}
        for call in message.tool_calls
    ]
    return {"role": message.role, "content": content}


def to_converse(request: LLMRequest, model_id: str) -> dict[str, Any]:
    """`converse(**kwargs)` for a request: system apart, tool results as user turns."""
    turns: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == "system":
            continue
        if message.role != "tool":
            turns.append(_turn(message))
            continue
        result = {
            "toolResult": {
                "toolUseId": message.tool_call_id,
                "content": [{"text": message.content}],
            }
        }
        if turns and turns[-1]["role"] == "user" and "toolResult" in turns[-1]["content"][0]:
            turns[-1]["content"].append(result)  # results of one turn share one user message
        else:
            turns.append({"role": "user", "content": [result]})
    kwargs: dict[str, Any] = {"modelId": model_id, "messages": turns}
    system = [{"text": m.content} for m in request.messages if m.role == "system"]
    if system:
        kwargs["system"] = system
    config = {"maxTokens": request.max_output_tokens, "temperature": request.temperature,
              "topP": request.top_p, "stopSequences": list(request.stop) or None}  # fmt: skip
    if any(value is not None for value in config.values()):
        kwargs["inferenceConfig"] = {k: v for k, v in config.items() if v is not None}
    if request.tools and request.tool_choice != "none":  # Converse has no "none": send no tools
        specs = [
            {
                "toolSpec": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": {"json": t.parameters},
                }
            }
            for t in request.tools
        ]
        choice: dict[str, Any] = {"auto": {"auto": {}}, "required": {"any": {}}}.get(
            request.tool_choice, {"tool": {"name": request.tool_choice}}
        )
        kwargs["toolConfig"] = {"tools": specs, "toolChoice": choice}
    return kwargs


def from_converse(raw: dict[str, Any]) -> dict[str, Any]:
    """A `CallableModel` reply mapping from a Converse response."""
    blocks = raw.get("output", {}).get("message", {}).get("content", [])
    calls = [
        ToolCall(
            id=b["toolUse"]["toolUseId"],
            name=b["toolUse"]["name"],
            arguments=b["toolUse"].get("input") or {},
        )
        for b in blocks
        if "toolUse" in b
    ]
    usage = raw.get("usage", {})
    reason = raw.get("stopReason", "")
    return {
        "text": "".join(b["text"] for b in blocks if "text" in b),
        "tool_calls": calls,
        "finish_reason": STOP.get(reason, "other"),
        "raw_finish_reason": reason,
        "usage": Usage(  # Converse counts cache reads and writes apart from inputTokens
            input_tokens=usage.get("inputTokens", 0),
            cached_input_tokens=usage.get("cacheReadInputTokens", 0),
            cache_write_input_tokens=usage.get("cacheWriteInputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        ),
    }


# --8<-- [end:convert]


# --8<-- [start:model]
def converse_model(client: Any, model_id: str) -> CallableModel:
    """A model port over a `bedrock-runtime` client's `converse` call."""

    def call(request: LLMRequest) -> dict[str, Any]:
        model = request.model or model_id
        try:
            raw = client.converse(**to_converse(request, model))  # sync: runs in a worker thread
        except Exception as exc:
            error = typed_error(exc, provider="bedrock", model=model)
            if error is None:
                raise
            raise error from exc
        return from_converse(raw)

    return CallableModel(call, provider="bedrock", model=model_id)


def bedrock_model(model_id: str, region: str = "us-east-1") -> CallableModel:
    """Bedrock with the default AWS credential chain; boto3 is imported on first use."""
    import boto3

    return converse_model(boto3.client("bedrock-runtime", region_name=region), model_id)


# --8<-- [end:model]

# --8<-- [start:prices]
PRICES = Prices.load(  # example rates: copy yours from the Bedrock price page for your region
    {
        "version": "mine-2026-11",
        "models": {"amazon.nova-pro-v1:0": {"provider": "bedrock", "input": 0.80, "output": 3.20}},
    }
)
# --8<-- [end:prices]


class FakeBedrock:
    """Stands in for `boto3.client("bedrock-runtime")`: asks for one tool, then answers."""

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        if tool_turn(kwargs["messages"]):
            status = kwargs["messages"][-1]["content"][0]["toolResult"]["content"][0]["text"]
            content: list[dict[str, Any]] = [{"text": f"Ticket T-1: {status}."}]
            reason = "end_turn"
        else:
            use = {"toolUseId": "tooluse_1", "name": "ticket_status", "input": {"ticket": "T-1"}}
            content, reason = [{"toolUse": use}], "tool_use"
        return {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": reason,
            "usage": {"inputTokens": 900, "outputTokens": 25, "totalTokens": 925},
        }


async def main() -> None:
    await demo(converse_model(FakeBedrock(), "amazon.nova-pro-v1:0"), PRICES)


if __name__ == "__main__":
    asyncio.run(main())
