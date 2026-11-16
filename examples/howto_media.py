"""Media in tool results: a tool returns an image; each provider gets what it can take."""

import asyncio

from agent_adapter import (
    CallableModel,
    Executor,
    LLMRequest,
    Media,
    Message,
    Registry,
    run_tool_loop,
)
from agent_adapter.schema.media import anthropic_content, openai_chat_content, openai_output

registry = Registry()
PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 2048  # stands in for real screenshot bytes


# --8<-- [start:tool]
@registry.tool
def screenshot(url: str) -> list[str | Media]:
    """Screenshot a web page."""
    return [f"screenshot of {url}", Media(data=PNG, name="page.png")]  # MIME type sniffed


# Also: Media.from_file("chart.pdf") or Media(url="https://example.com/cat.png")
# --8<-- [end:tool]


# --8<-- [start:loop]
def my_llm(request: LLMRequest) -> dict | str:  # your model; the tool message holds the media
    last = request.messages[-1]
    if last.role == "tool":
        return f"{last.content}: {last.media[0].describe()}"
    return {"tool_calls": [{"name": "screenshot", "arguments": {"url": "https://example.com"}}]}


async def main() -> None:
    ask = [Message(role="user", content="What is on example.com?")]
    result = await run_tool_loop(CallableModel(my_llm), registry, ask, executor=Executor(registry))
    print(result.final.text)


# --8<-- [end:loop]

# --8<-- [start:rewrites]
shot = Media(data=PNG, name="page.png")
song = Media(data=b"ID3" + b"\0" * 64, name="song.mp3")
print([item["type"] for item in openai_output("", [shot, song])])  # Responses API
print([block["type"] for block in anthropic_content("", [shot])])
print(openai_chat_content("done", [shot]))  # Chat Completions: tool messages are text only
# --8<-- [end:rewrites]

if __name__ == "__main__":
    asyncio.run(main())
