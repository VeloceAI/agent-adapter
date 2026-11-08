"""Shared helpers for the memory tests: scopes, records, a tiny topic embedder."""

from datetime import UTC, datetime, timedelta

from agent_adapter.embeddings import CallableEmbedder
from agent_adapter.memory import MemoryRecord, MemoryScope

ALICE = MemoryScope(user_id="alice", session_id="s1")
ALICE_S2 = MemoryScope(user_id="alice", session_id="s2")
ALICE_ALL = MemoryScope(user_id="alice")
BOB = MemoryScope(user_id="bob", session_id="s1")
BOB_ALL = MemoryScope(user_id="bob")

_TOPICS = {
    "pets": {"cat", "cats", "kitten", "dog", "puppy", "pet"},
    "food": {"pizza", "pasta", "vegan", "lunch", "food"},
    "travel": {"train", "flight", "museum", "trip", "ticket"},
}


def topics(texts: list[str]) -> list[list[float]]:
    """A fake embedding: one dimension per topic, counting its words."""
    rows = []
    for text in texts:
        words = set(text.lower().replace(".", " ").split())
        rows.append([float(len(words & vocab)) + 0.01 for vocab in _TOPICS.values()])
    return rows


def embedder() -> CallableEmbedder:
    return CallableEmbedder(topics, model="topics-v1", provider="local")


def rec(scope: MemoryScope, text: str, **kw) -> MemoryRecord:
    return MemoryRecord(scope=scope, text=text, **kw)


def days_ago(days: float) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)
