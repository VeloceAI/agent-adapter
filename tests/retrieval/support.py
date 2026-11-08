"""Shared fakes for the retrieval tests: a keyword embedder and small corpora."""

from agent_adapter.embeddings import CallableEmbedder
from agent_adapter.retrieval import Document, Source

VOCAB = ("ticket", "refund", "museum", "opening", "hours", "parking", "shop")


def embed(texts: list[str]) -> list[list[float]]:
    """One count per vocabulary word: texts sharing words point the same way."""
    return [[float(text.lower().count(word)) for word in VOCAB] for text in texts]


def embedder() -> CallableEmbedder:
    """A deterministic offline embedder."""
    return CallableEmbedder(embed, model="keywords-v1")


def doc(id: str, text: str, **metadata: object) -> Document:
    """A document with a source built from its id."""
    return Document(
        id=id,
        text=text,
        source=Source(uri=f"https://help.example.com/{id}", title=id.title()),
        metadata=dict(metadata),
    )


CORPUS = (
    doc("refunds", "A ticket refund is possible up to 24 hours before the visit.", scope="north"),
    doc("hours", "The museum opening hours are 9 to 17, every day.", scope="north"),
    doc("parking", "Parking is free for museum visitors.", scope="south"),
    doc("shop", "The shop sells a ticket bundle with parking.", scope="south"),
)
