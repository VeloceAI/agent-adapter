"""LangChain alone: compress the message list before `model.invoke`; no LangGraph needed."""

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent_adapter.context import CompressionPolicy, RemoveOldMessages, compress_for_sync

OLD = "an old turn with a lot of detail " * 30
policy = CompressionPolicy(
    start_when_tokens_over=1_000, shrink_to_tokens=600, steps=[RemoveOldMessages(keep_last=2)]
)
llm = FakeListChatModel(responses=["The rate is 30%."])  # your chat model


# --8<-- [start:langchain]
def ask(messages: list[Any]) -> Any:
    smaller, _ = compress_for_sync(messages, policy)  # BaseMessage in, BaseMessage out
    return llm.invoke(smaller)


# --8<-- [end:langchain]


def history() -> list[Any]:
    turns: list[Any] = [SystemMessage(content="Be brief.")]
    for n in range(3):
        turns += [HumanMessage(content=f"q{n} {OLD}"), AIMessage(content=f"a{n} {OLD}")]
    return [*turns, HumanMessage(content="What is the rate?")]


def main() -> None:
    messages = history()
    smaller, _ = compress_for_sync(messages, policy)
    print("langchain:", len(messages), "messages,", len(smaller), "sent")
    print("reply:", ask(messages).content)


if __name__ == "__main__":
    main()
