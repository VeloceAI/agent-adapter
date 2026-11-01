"""Compress a framework's own message list where it calls the model; no framework installed."""

import asyncio
import dataclasses
from types import SimpleNamespace
from typing import Any

from agent_adapter.context import (
    CompressionPolicy,
    RemoveOldMessages,
    compress_for,
    compress_for_sync,
)

OLD = "an old turn with a lot of detail " * 30
policy = CompressionPolicy(
    start_when_tokens_over=1_000, shrink_to_tokens=600, steps=[RemoveOldMessages(keep_last=2)]
)


# --8<-- [start:openai]
def ask_openai(client: Any, messages: list[dict]) -> Any:
    smaller, _ = compress_for_sync(messages, policy)  # chat dicts in, chat dicts out
    return client.chat.completions.create(model="gpt-4.1", messages=smaller)


# --8<-- [end:openai]


# --8<-- [start:anthropic]
def ask_anthropic(client: Any, system: str, messages: list[dict]) -> Any:
    smaller, _ = compress_for_sync(messages, policy, format="anthropic")
    return client.messages.create(
        model="claude-sonnet-4-5", system=system, messages=smaller, max_tokens=1024
    )


# --8<-- [end:anthropic]


# --8<-- [start:gemini]
def ask_gemini(client: Any, contents: list[Any]) -> Any:
    smaller, _ = compress_for_sync(contents, policy, format="gemini")  # dicts or types.Content
    return client.models.generate_content(model="gemini-2.5-flash", contents=smaller)


# --8<-- [end:gemini]


# --8<-- [start:openai_agents]
async def compress_model_input(data: Any) -> Any:
    """RunConfig(call_model_input_filter=compress_model_input): Responses items each turn."""
    smaller, _ = await compress_for(data.model_data.input, policy, format="openai")
    return dataclasses.replace(data.model_data, input=smaller)


# --8<-- [end:openai_agents]


# --8<-- [start:adk]
def before_model(callback_context: Any, llm_request: Any) -> None:
    """LlmAgent(before_model_callback=before_model): shorten what this model call sends."""
    llm_request.contents, _ = compress_for_sync(llm_request.contents, policy, format="gemini")


# --8<-- [end:adk]


class _Recorder:
    """Stands in for an SDK client: records what would be sent."""

    def __init__(self) -> None:
        self.sent: list[Any] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._send))
        self.messages = SimpleNamespace(create=self._send)
        self.models = SimpleNamespace(generate_content=self._send)

    def _send(self, **kwargs: Any) -> int:
        payload = kwargs.get("messages", kwargs.get("contents"))
        self.sent.append(payload)
        return len(payload)


@dataclasses.dataclass
class _ModelInputData:  # the shape of agents.run.ModelInputData
    input: list[Any]
    instructions: str | None


def main() -> None:
    client = _Recorder()
    chat = [{"role": "system", "content": "Be brief."}]
    chat += [{"role": r, "content": OLD} for r in ("user", "assistant") * 3]
    chat.append({"role": "user", "content": "And now?"})
    print("openai:", len(chat), "->", ask_openai(client, chat))
    claude = [{"role": r, "content": OLD} for r in ("user", "assistant") * 3]
    claude.append({"role": "user", "content": "And now?"})
    print("anthropic:", len(claude), "->", ask_anthropic(client, "Be brief.", claude))
    contents = [{"role": r, "parts": [{"text": OLD}]} for r in ("user", "model") * 3]
    contents.append({"role": "user", "parts": [{"text": "And now?"}]})
    print("gemini:", len(contents), "->", ask_gemini(client, contents))
    data = SimpleNamespace(model_data=_ModelInputData(input=chat[1:], instructions="Be brief."))
    filtered = asyncio.run(compress_model_input(data))
    print("openai agents:", len(chat) - 1, "->", len(filtered.input), filtered.instructions)
    request = SimpleNamespace(contents=list(contents))
    before_model(None, request)
    print("adk:", len(contents), "->", len(request.contents))


if __name__ == "__main__":
    main()
