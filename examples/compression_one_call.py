"""One model call, four policies: trim only, cut only, summarise only, and all combined."""

import asyncio
import json

from agent_adapter import CallableModel, Ledger, LLMRequest, MemoryCache, Message, Usage
from agent_adapter.context import (
    CallableReranker,
    CompressionPolicy,
    CutLongMessages,
    KeepMostRelevant,
    RemoveOldMessages,
    RerankerScorer,
    RerankResult,
    ShrinkToolResults,
    Summarize,
    SummarizeLongMessage,
)

TURN = "the client asked about refunds and we went through each order line " * 10


def history() -> tuple[Message, ...]:
    turns = [Message(role="system", content="You answer help requests.")]
    for n in range(4):
        turns.append(Message(role="user", content=f"question {n}: {TURN}"))
        turns.append(Message(role="assistant", content=f"answer {n}: {TURN}"))
    turns.append(Message(role="user", content="So what can I claim?"))
    return tuple(turns)


def main_llm(request: LLMRequest) -> str:
    """Your model; here it says how much it was sent."""
    chars = sum(len(m.content) for m in request.messages)
    return f"{len(request.messages)} messages, {chars} characters"


def summary_llm(request: LLMRequest) -> str:
    """Your (cheaper) summary model; it must answer in the requested JSON shape."""
    return json.dumps({"summary": "Earlier turns covered the refund.", "facts": ["30d"]})


model = CallableModel(main_llm, provider="local", model="main")
summariser = CallableModel(summary_llm, provider="local", model="small")
LIMITS = {"start_when_tokens_over": 800, "shrink_to_tokens": 500}  # or the percents

# --8<-- [start:trim]
trim_only = CompressionPolicy(**LIMITS, steps=[RemoveOldMessages(keep_first=0, keep_last=2)])
# --8<-- [end:trim]
# --8<-- [start:cut]
cut_only = CompressionPolicy(**LIMITS, steps=[CutLongMessages(max_tokens_per_message=60)])
# --8<-- [end:cut]
# --8<-- [start:summarise]
summarise_only = CompressionPolicy(
    **LIMITS,
    steps=[Summarize(model=summariser, keep_recent_messages=2)],
    cache=MemoryCache(),  # the same turns are never summarised (or paid for) twice
)
# --8<-- [end:summarise]
# --8<-- [start:combined]
combined = CompressionPolicy(
    **LIMITS,
    steps=[
        ShrinkToolResults(),  # free: old, big tool results become a short note
        CutLongMessages(max_tokens_per_message=100),  # free: long messages are cut
        Summarize(model=summariser, keep_recent_messages=4),  # one call to your summariser
    ],
    if_still_too_long=CompressionPolicy(**LIMITS, steps=[RemoveOldMessages(keep_last=2)]),
)
# --8<-- [end:combined]


# --8<-- [start:call]
async def ask(policy: CompressionPolicy) -> str:
    reply = await model.generate(LLMRequest(messages=history(), compression=policy))
    return reply.text


# --8<-- [end:call]


# --8<-- [start:rerank]
def my_reranker(query: str, texts: list[str]) -> RerankResult:
    """Your reranker (an API or a local cross-encoder); scores in 0..1, one per text."""
    scores = [1.0 if "3" in text else 0.1 for text in texts]
    return RerankResult(scores=scores, usage=Usage(units=1))  # usage is what gets billed


ledger = Ledger()
reranker = CallableReranker(my_reranker, provider="my-rerank", model="v1", ledger=ledger)
relevant_only = CompressionPolicy(
    **LIMITS, steps=[KeepMostRelevant(scorer=RerankerScorer(reranker), keep_count=2)]
)
# --8<-- [end:rerank]

REFERENCE = "Article 12A allows free returns for unused purchases. " * 60

# --8<-- [start:split]
split = (
    Message(role="system", content="You answer help requests. Cite the article."),  # rules only
    Message(role="user", content=f"Reference material:\n{REFERENCE}"),  # may be cut or summarised
    Message(role="user", content="So what can I claim?"),
)
# --8<-- [end:split]

# --8<-- [start:system]
rules_and_reference = (
    "You answer help requests. Cite the article.\n"  # rules: always kept word for word
    f"<reference>\n{REFERENCE}\n</reference>"  # only this block may be summarised
)
big_system = CompressionPolicy(
    **LIMITS,
    steps=[
        SummarizeLongMessage(
            model=summariser,
            bigger_than_tokens=500,
            include_system_prompt=True,  # may shorten the system text; never removes it
            instructions="Keep article numbers.",
        ),
    ],
)
# --8<-- [end:system]


async def main() -> None:
    print("before:", main_llm(LLMRequest(messages=history())))
    for name, policy in (
        ("trim", trim_only),
        ("cut", cut_only),
        ("summarise", summarise_only),
        ("combined", combined),
    ):
        print(f"{name}:", await ask(policy))
    print("relevant:", await ask(relevant_only))
    entry = ledger.entries()[-1]
    print("rerank entry:", entry.kind, entry.stage, entry.units)
    print("split:", (await model.generate(LLMRequest(messages=split, compression=cut_only))).text)
    big = (Message(role="system", content=rules_and_reference), split[-1])
    print("system:", (await model.generate(LLMRequest(messages=big, compression=big_system))).text)


if __name__ == "__main__":
    asyncio.run(main())
