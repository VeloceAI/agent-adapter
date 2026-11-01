"""Offline stand-ins for the agent model and the proposer model.

Delete this file in real use and pass `OpenAIModel(...)`, `AnthropicModel(...)` or any
`LLMPort` instead.
"""

import json

from agent_adapter import CallableModel, LLMRequest, LLMResponse, Message, Prices, ToolCall

FREE = Prices.load({"version": "fakes", "models": {"fake": {"provider": "fake", "free": True}}})
MEMORY = {"return an item": "HC-101", "refund status": "HC-102"}  # what it knows unaided


def _text(content: str) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="fake",
        text=content,
        message=Message(role="assistant", content=content),
    )


async def _faq_answer(request: LLMRequest) -> LLMResponse:
    """Driven only by harness text: it looks a topic up when told to, cites when told to."""
    system = request.messages[0].content.lower()
    must_look_up = "lookup_article first" in system or "for every topic" in system
    cites = "cite the article" in system
    last = request.messages[-1]
    if last.role == "tool":
        found = last.content
        return _text(f"Article {found}." if cites else "I checked the help center; it may help.")
    topic = last.content
    if topic in MEMORY:
        return _text(f"Article {MEMORY[topic]}." if cites else f"That is {MEMORY[topic]}.")
    if not must_look_up:
        return _text("An article may help.")
    call = ToolCall(name="lookup_article", arguments={"topic": topic}, id="call_1")
    return LLMResponse(
        provider="fake",
        model="fake",
        tool_calls=(call,),
        finish_reason="tool_calls",
        message=Message(role="assistant", tool_calls=(call,)),
    )


async def _proposal(request: LLMRequest) -> LLMResponse:
    """Scripted edits by stage: the knowledge editor asks for lookups and citations."""
    edits = {
        "propose:know1": {
            "path": "prompt/system",
            "content": "You answer customer-support questions. "
            "Always call lookup_article first, then cite the article id.",
            "reason": "6 of 8 answers name no article and no lookup was made",
        },
        "propose:proc1": {
            "path": "tool/lookup_article/prompt_hint",
            "content": "Call this for every topic before answering.",
            "reason": "the tool was never called",
        },
    }
    edit = edits.get(request.stage)
    if edit is None:
        reply = {"keep_base": True, "notes": "nothing to change", "edits": []}
    else:
        reply = {
            "keep_base": False,
            "notes": "require a lookup",
            "edits": [{**edit, "delete": False}],
        }
    return _text(json.dumps(reply))


class FakeFaqModel(CallableModel):
    """The agent's model: deterministic, offline."""

    def __init__(self) -> None:
        super().__init__(_faq_answer, provider="fake", model="faq-agent")


class ScriptedProposer(CallableModel):
    """The proposer model: one scripted edit per editor."""

    def __init__(self) -> None:
        super().__init__(_proposal, provider="fake", model="proposer")
