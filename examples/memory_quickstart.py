"""Memory: a ticket desk agent saves a seat preference in one chat and recalls it in the next.

A scripted model stands in for a real one, so it runs with no network and no API key.
"""

import asyncio
import dataclasses
import json
import tempfile
from pathlib import Path

from agent_adapter import (
    CallableModel,
    Executor,
    LLMRequest,
    LLMResponse,
    Message,
    Registry,
    ToolCall,
    run_tool_loop,
)
from agent_adapter.memory import MemoryScope, SQLiteMemory, memory_tools


def scripted(request: LLMRequest) -> LLMResponse:
    """Saves "remember ..." messages, looks up everything else, then answers from the result."""
    asked = request.messages[-1]
    if asked.role == "tool":
        if asked.name == "remember":
            text = "I will remember that."
        else:
            facts = [item["text"] for item in json.loads(asked.content)]
            text = f"From earlier chats: {facts[0]}" if facts else "No preference saved."
        message = Message(role="assistant", content=text)
        return LLMResponse(provider="local", model="scripted", text=text, message=message)
    words = asked.content.split(maxsplit=1)
    if words[0].lower() == "remember":
        call = ToolCall(name="remember", arguments={"text": words[1]}, id="call_1")
    else:
        call = ToolCall(name="recall", arguments={"query": "seat"}, id="call_1")
    message = Message(role="assistant", tool_calls=(call,))
    return LLMResponse(provider="local", model="scripted", tool_calls=(call,), message=message)


# --8<-- [start:wire]
def build(folder: Path) -> tuple[Executor, Registry, SQLiteMemory]:
    """One memory file, its three tools in a registry, and an executor that runs them."""
    memory = SQLiteMemory(folder / "memory.db", retention_days=90)
    registry = Registry()
    registry.add_many(memory_tools(memory))  # remember, recall, forget
    return Executor(registry), registry, memory


async def chat(executor: Executor, registry: Registry, user: str, chat_id: str, text: str) -> str:
    """One turn. The host says who is talking; the model never chooses the scope."""
    ctx = dataclasses.replace(executor.context(thread_id=chat_id), user_id=user)
    model = CallableModel(scripted, provider="local", model="scripted")
    messages = [Message(role="user", content=text)]
    result = await run_tool_loop(model, registry, messages, executor=executor, ctx=ctx)
    return result.final.text or ""


# --8<-- [end:wire]


async def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        executor, registry, memory = build(Path(folder))
        print(await chat(executor, registry, "ann", "chat-1", "remember Ann prefers aisle seats"))
        print(await chat(executor, registry, "ann", "chat-2", "Which seat should I book for Ann?"))
        print(await chat(executor, registry, "ben", "chat-3", "Which seat should I book for Ben?"))
        # --8<-- [start:direct]
        hits = await memory.search("aisle", scope=MemoryScope(user_id="ann"))
        print([hit.record.text for hit in hits])
        # --8<-- [end:direct]


if __name__ == "__main__":
    asyncio.run(main())
