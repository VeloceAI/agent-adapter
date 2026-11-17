"""Sources as short passages: split documents, rank passages for a blank (BM25), or retrieve."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from agent_adapter.core.context import ToolContext
from agent_adapter.retrieval.documents import Document
from agent_adapter.retrieval.port import RetrieverPort

PASSAGE_CHARS = 600
TOP_K = 4
_WORD = re.compile(r"[a-z0-9]+")
_BREAK = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


@dataclass(frozen=True)
class Passage:
    """A piece of one source's text, cut at sentence ends; cited by its source's id."""

    source: Document
    text: str


def passages(source: Document, size: int = PASSAGE_CHARS) -> list[Passage]:
    """`source` cut into passages of at most about `size` characters."""
    found: list[Passage] = []
    current = ""
    for sentence in _BREAK.split(source.text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > size:
            if current:
                found.append(Passage(source, current))
                current = ""
            found.append(Passage(source, sentence[:size]))
            sentence = sentence[size:]
        if current and len(current) + 1 + len(sentence) > size:
            found.append(Passage(source, current))
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        found.append(Passage(source, current))
    return found


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def rank(query: str, pool: Sequence[Passage], k: int = TOP_K) -> list[Passage]:
    """The `k` passages that best match `query` by BM25 word overlap, best first."""
    if not pool or k < 1:
        return []
    terms = set(words(query))
    docs = [Counter(words(p.text)) for p in pool]
    average = sum(sum(d.values()) for d in docs) / len(docs) or 1.0
    seen = Counter(term for d in docs for term in d if term in terms)
    scores = []
    for index, counts in enumerate(docs):
        length = sum(counts.values())
        score = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (len(docs) - seen[term] + 0.5) / (seen[term] + 0.5))
            score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / average))
        scores.append((score, -index))
    best = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
    return [pool[i] for i in best[:k] if scores[i][0] > 0] or [pool[i] for i in best[:k]]


async def select(
    query: str,
    sources: Sequence[Document],
    *,
    k: int = TOP_K,
    retriever: RetrieverPort | None = None,
    ctx: ToolContext | None = None,
) -> list[Passage]:
    """The passages a blank sees: from `retriever`, else the best `k` of its sources."""
    if retriever is not None:
        found = await retriever.retrieve(query, k=k, ctx=ctx)
        return [Passage(doc, doc.text[:PASSAGE_CHARS]) for doc in found]
    pool = [passage for source in sources for passage in passages(source)]
    return rank(query, pool, k)


def window(text: str, phrase: str, size: int = 300) -> str:
    """About `size` characters of `text` around where `phrase` matches it best."""
    if len(text) <= size:
        return text
    wanted = set(words(phrase))
    step = max(size // 3, 1)
    best, best_score = 0, -1
    for start in range(0, max(len(text) - size, 0) + step, step):
        score = len(wanted & set(words(text[start : start + size])))
        if score > best_score:
            best, best_score = start, score
    return text[best : best + size]
