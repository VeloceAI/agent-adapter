"""No agent framework: a script and a FastAPI route call your model with compression on."""

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_adapter import CallableModel, LLMRequest, Message, Registry, run_tool_loop
from agent_adapter.context import CompressingModel, CompressionPolicy

TURN = "a long answer about shipping fees " * 20  # about 170 tokens per old turn
SYSTEM = Message(role="system", content="You answer help requests.")


def long_history() -> list[Message]:
    """A system prompt, 12 old turns and the new question."""
    history = [SYSTEM]
    for n in range(6):
        history.append(Message(role="user", content=f"question {n}: {TURN}"))
        history.append(Message(role="assistant", content=f"answer {n}: {TURN}"))
    history.append(Message(role="user", content="What is the euro rate?"))
    return history


# --8<-- [start:model]
def my_llm(request: LLMRequest) -> str:
    """Your own model call: an HTTP API, a local model... Here it says what it was sent."""
    chars = sum(len(m.content) for m in request.messages)
    return f"{len(request.messages)} messages, {chars} characters"


model = CallableModel(my_llm, provider="local", model="my-model")
# --8<-- [end:model]

# --8<-- [start:script]
policy = CompressionPolicy(
    model_max_tokens=4_000,  # your model's context window, in tokens
    start_when_full_percent=50,  # compress once the request fills half of it...
    shrink_to_percent=40,  # ...down to 40%
)


async def answer(history: list[Message]) -> str:
    request = LLMRequest(messages=tuple(history), compression=policy)
    reply = await model.generate(request)  # the oldest turns are dropped first
    return reply.text


# --8<-- [end:script]


# --8<-- [start:fastapi]
class ChatTurn(BaseModel):
    """A turn as a client may send it: no system role; `pinned`, `raw` and the rest dropped."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)  # characters per turn


class ChatBody(BaseModel):
    messages: list[ChatTurn] = Field(min_length=1, max_length=200)


def make_app() -> Any:
    from fastapi import FastAPI  # imported only when the app is built

    app = FastAPI()

    @app.post("/chat")
    async def chat(body: ChatBody) -> dict[str, str]:
        turns = [Message(role=turn.role, content=turn.content) for turn in body.messages]
        return {"reply": await answer([SYSTEM, *turns])}  # your system prompt, not theirs

    return app


# --8<-- [end:fastapi]

# --8<-- [start:loop]
registry = Registry()


@registry.tool
def exchange_rate(currency: str) -> str:
    """Today's EUR rate for a currency."""
    return f"{currency}: 0.92"


async def agent(history: list[Message]) -> str:
    """The adapter as the agent loop: your model, your tools, compression before each call."""
    result = await run_tool_loop(model, registry, history, compression=policy)
    return result.final.text


every_call = CompressingModel(model, policy)  # or compress every call of one model
# --8<-- [end:loop]


async def main() -> None:
    history = long_history()
    print(my_llm(LLMRequest(messages=tuple(history))))
    print(await answer(history))
    print(await agent(history))
    print((await every_call.generate(LLMRequest(messages=tuple(history)))).text)


if __name__ == "__main__":
    asyncio.run(main())
