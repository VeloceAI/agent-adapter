"""`KnownNames`: fix misheard names in a transcript against a list the developer knows."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

_WORD = re.compile(r"[A-Za-z][A-Za-z']*")
_SOUNDS = (
    ("ph", "f"), ("ck", "k"), ("q", "k"), ("c", "k"), ("z", "s"), ("x", "ks"), ("w", "v"),
    ("y", "i"), ("ee", "i"), ("oo", "u"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"),
    ("gh", "g"), ("sh", "s"),
)  # fmt: skip
_COMMON_WORDS = (
    "the", "and", "that", "this", "with", "have", "from", "they", "will", "would", "there",
    "their", "what", "about", "which", "when", "make", "like", "time", "just", "know", "take",
    "into", "year", "your", "good", "some", "could", "them", "other", "than", "then", "look",
    "only", "come", "over", "think", "also", "back", "after", "work", "first", "well", "even",
    "want", "because", "these", "give", "most", "very", "call", "hello", "yes", "yeah", "okay",
    "sure", "please", "thanks", "thank", "sorry", "right", "maybe", "need", "help", "today",
    "tomorrow", "order", "account", "money", "name", "number", "phone", "email", "address",
)  # fmt: skip
COMMON = frozenset(_COMMON_WORDS)


@dataclass(frozen=True)
class NameFix:
    """One change: what was heard, the name it became, and how close they were (0 to 1)."""

    heard: str
    name: str
    score: float


def sound_key(word: str) -> str:
    """A rough sound-alike key: "Rahool" and "Rahul" both give "rl"."""
    word = word.lower()
    for written, said in _SOUNDS:
        word = word.replace(written, said)
    head, tail = word[:1], word[1:]
    skeleton = "".join(c for c in tail if c not in "aeiouh'")
    key = head
    for c in skeleton:
        if c != key[-1:]:
            key += c
    return key


def distance(a: str, b: str, limit: int) -> int:
    """Edit distance, giving up (returning `limit + 1`) once it must exceed `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


class KnownNames:
    """Fixes words that sound like a known name: "I'm Rahool from Acmy Corp" ->
    "I'm Rahul from Acme Corp", for `KnownNames(["Rahul", "Acme Corp"])`.

    A word changes when its spelling is at least `cutoff` alike (1 minus edit distance over
    length), or `sound_cutoff` alike and it sounds the same. Words shorter than `min_len`
    and common English words never change; exact matches only get the name's capitals.
    `last_fixes` lists the guesses of the latest call; guessed names deserve a read-back.
    """

    def __init__(
        self,
        names: Iterable[str],
        *,
        cutoff: float = 0.8,
        sound_cutoff: float = 0.65,
        min_len: int = 4,
        common: frozenset[str] = COMMON,
    ) -> None:
        self.cutoff = cutoff
        self.sound_cutoff = sound_cutoff
        self.min_len = min_len
        self.common = common
        self.names = [n.strip() for n in names if n.strip()]
        self.last_fixes: list[NameFix] = []
        self._exact = {n.lower(): n for n in self.names}
        self._by_first: dict[tuple[int, str], list[str]] = defaultdict(list)
        self._by_sound: dict[tuple[int, str], list[str]] = defaultdict(list)
        for name in self.names:
            words = len(name.split())
            low = name.lower()
            self._by_first[words, low[0]].append(low)
            self._by_sound[words, sound_key(low.replace(" ", ""))].append(low)
        self._sizes = sorted({len(n.split()) for n in self.names}, reverse=True)
        self._memo: dict[str, NameFix | None] = {}

    def __call__(self, text: str) -> str:
        return self.fix(text)

    def fix(self, text: str) -> str:
        """`text` with misheard names replaced; `last_fixes` says what changed."""
        self.last_fixes = []
        if not self.names:
            return text
        spans = [(m.start(), m.end(), m[0]) for m in _WORD.finditer(text)]
        out, cursor, i = [], 0, 0
        while i < len(spans):
            for size in self._sizes:
                if i + size > len(spans):
                    continue
                start, end = spans[i][0], spans[i + size - 1][1]
                heard = " ".join(word for _, _, word in spans[i : i + size])
                if size > 1 and not text[start:end].replace(" ", "").isalpha():
                    continue  # punctuation between the words: not one name
                found = self._match(heard, size)
                if found is not None:
                    out.append(text[cursor:start] + found.name)
                    cursor, i = end, i + size
                    if found.score < 1:
                        self.last_fixes.append(found)
                    break
            else:
                i += 1
        out.append(text[cursor:])
        return "".join(out)

    def _match(self, heard: str, size: int) -> NameFix | None:
        key = f"{size}:{heard}"
        if key in self._memo:
            return self._memo[key]
        low = heard.lower()
        exact = self._exact.get(low)
        if exact is not None:
            found: NameFix | None = NameFix(heard, exact, 1.0)
        elif len(low.replace(" ", "")) < self.min_len or (size == 1 and low in self.common):
            found = None
        else:
            found = self._closest(heard, low, size)
        if len(self._memo) > 4096:
            self._memo.clear()
        self._memo[key] = found
        return found

    def _closest(self, heard: str, low: str, size: int) -> NameFix | None:
        sound = sound_key(low.replace(" ", ""))
        alike = set(self._by_sound.get((size, sound), ()))
        best: NameFix | None = None
        limit = int(len(low) * (1 - self.sound_cutoff))
        for name in alike | set(self._by_first.get((size, low[0]), ())):
            longest = max(len(name), len(low))
            score = 1 - distance(low, name, limit) / longest
            needed = self.sound_cutoff if name in alike else self.cutoff
            if score >= needed and (best is None or score > best.score):
                best = NameFix(heard, self._exact[name], round(score, 3))
        return best
