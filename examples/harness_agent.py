"""The agent being optimised: a help-center FAQ agent with one tool and no framework."""

from agent_adapter import LLMRequest, Message, ToolDef
from agent_adapter.models import LLMPort
from agent_adapter.optimiser import EvalCase, HarnessTree

ARTICLES = {
    "return an item": "HC-101",
    "refund status": "HC-102",
    "shipping times": "HC-201",
    "track a parcel": "HC-202",
    "warranty claim": "HC-301",
    "reset my password": "HC-401",
    "download an invoice": "HC-501",
    "cancel a subscription": "HC-502",
}


def lookup_article(topic: str) -> str:
    """The id of the help-center article that covers a topic."""
    return ARTICLES.get(topic, "unknown")


TOOLS = {"lookup_article": lookup_article}
PARAMETERS = {
    "type": "object",
    "properties": {"topic": {"type": "string"}},
    "required": ["topic"],
    "additionalProperties": False,
}

# --8<-- [start:surface]
SURFACE = {  # the texts the optimiser may tune, and their base values
    "prompt/system": "You answer customer-support questions.",
    "tool/lookup_article/description": "Look up the help-center article for a topic.",
    "tool/lookup_article/prompt_hint": "Use it when you are unsure.",
}
# --8<-- [end:surface]


# --8<-- [start:agent]
async def run_case(tree: HarnessTree, case: EvalCase, model: LLMPort) -> str:
    """One answer. Swap in your own agent here: read the texts you want tuned from `tree`."""
    files = tree.files
    system = files["prompt/system"]
    if hint := files.get("tool/lookup_article/prompt_hint"):
        system += f"\n\nWhen to use lookup_article: {hint}"
    tool = ToolDef(
        name="lookup_article",
        description=files["tool/lookup_article/description"],
        parameters=PARAMETERS,
    )
    messages = [
        Message(role="system", content=system),
        Message(role="user", content=str(case.input)),
    ]
    for _ in range(2):  # one tool round, then the answer
        request = LLMRequest(messages=tuple(messages), tools=(tool,), stage="agent")
        reply = await model.generate(request)  # use `model`: its calls are metered and cached
        if not reply.tool_calls:
            return reply.text
        messages.append(reply.message)
        for call in reply.tool_calls:
            result = TOOLS[call.name](**call.arguments)
            messages.append(
                Message(role="tool", content=result, tool_call_id=call.id, name=call.name)
            )
    raise RuntimeError("the agent asked for tools twice without answering")


# --8<-- [end:agent]
