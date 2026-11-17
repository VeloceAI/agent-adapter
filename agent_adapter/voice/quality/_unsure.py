"""`UnsureWords`: numbers, dates and names the speech-to-text was unsure of, read back before
a `confirm=True` tool runs."""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_adapter.core.types import ApprovalRequest
from agent_adapter.voice.quality._numbers import NUMBER_WORDS, words_number
from agent_adapter.voice.quality._spoken import MONTHS

_MONTH_OF = {m.lower(): i for i, m in enumerate(MONTHS, 1)}
_MONTH_OF |= {m[:3].lower(): i for i, m in enumerate(MONTHS, 1)}
_DIGITS = re.compile(r"\d[\d,]*(?:\.\d+)?")
_EDGE = ".,!?;:\"'()"


@dataclass(frozen=True)
class Word:
    """One heard word and the speech-to-text's confidence in it (0 to 1; None if not given)."""

    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class Unsure:
    """A number, date or name heard with low confidence."""

    text: str  # as heard: "4,500", "forty five hundred", "Rahul"
    kind: str  # "number", "date" or "name"
    confidence: float
    keys: tuple[str, ...]  # how it shows in tool arguments


def words_of(heard: Iterable[Any] | str) -> list[Word]:
    """`Word`s from a transcript, or from a provider's word list: `Word`s, or mappings with
    "word", "punctuated_word" or "text" and "confidence" (Deepgram, AssemblyAI, Soniox)."""
    if isinstance(heard, str):
        return [Word(token) for token in heard.split()]
    out = []
    for item in heard:
        if isinstance(item, Word):
            out.append(item)
        elif isinstance(item, Mapping):
            text = item.get("punctuated_word") or item.get("word") or item.get("text") or ""
            confidence = item.get("confidence")
            out.append(Word(str(text), None if confidence is None else float(confidence)))
        else:
            out.append(Word(str(item)))
    return out


def _number_key(raw: str) -> str:
    value = float(raw.replace(",", ""))
    return str(int(value)) if value.is_integer() else str(value)


class UnsureWords:
    """Keeps the low-confidence numbers, dates and names of each call's last `turns` turns,
    and reads them back before a `confirm=True` tool runs: "Did you say 4,500?".

    Feed it what the speech-to-text heard with `hear(words, run_id=...)` (a `check_stt`
    wrapper does this when the STT reports word confidence). Give `readback` to the spoken
    approval: `VoiceApproval(readback=unsure.readback)`. A word below `threshold` counts;
    with `match_arguments`, only the ones that show up in the tool's arguments are read back.
    Each is read back once. Words with no confidence are taken as sure.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.85,
        kinds: Iterable[str] = ("number", "date", "name"),
        names: Iterable[str] = (),
        turns: int = 3,
        max_age_s: float = 120.0,
        match_arguments: bool = True,
        template: str = "Did you say {items}?",
    ) -> None:
        self.threshold = threshold
        self.kinds = frozenset(kinds)
        self.names = frozenset(n.lower() for n in names)
        self.turns = turns
        self.max_age_s = max_age_s
        self.match_arguments = match_arguments
        self.template = template
        self._runs: dict[str | None, deque[tuple[float, list[Unsure]]]] = {}

    def hear(
        self,
        heard: Iterable[Any] | str,
        *,
        run_id: str | None = None,
        guessed: Iterable[tuple[str, float]] = (),
    ) -> list[Unsure]:
        """Note one turn's words; `guessed` are (name, score) a names fix put in. Returns the
        unsure items found."""
        found = [item for item in self._scan(words_of(heard)) if item.kind in self.kinds]
        if "name" in self.kinds:
            found += [Unsure(name, "name", score, (name.lower(),)) for name, score in guessed]
        turns = self._runs.setdefault(run_id, deque(maxlen=self.turns))
        turns.append((time.monotonic(), found))
        return found

    def unsure(self, run_id: str | None = None) -> list[Unsure]:
        """The unsure items still waiting for a read-back, oldest first."""
        turns = self._turns(run_id)
        if not turns:
            return []
        oldest = time.monotonic() - self.max_age_s
        return [item for at, items in turns if at >= oldest for item in items]

    def clear(self, run_id: str | None = None) -> None:
        self._runs.pop(run_id, None)

    def readback(self, request: ApprovalRequest) -> str | None:
        """The read-back line for this approval, or None when nothing needs one."""
        items = self.unsure(request.run_id)
        if self.match_arguments:
            numbers, texts = _argument_values(request.arguments)
            items = [i for i in items if _shows_in(i, numbers, texts)]
        chosen: dict[tuple[str, ...], Unsure] = {}
        for item in items:
            chosen.setdefault(item.keys, item)
        if not chosen:
            return None
        self._forget(request.run_id, set(chosen.values()))
        said = [item.text for item in chosen.values()]
        joined = said[0] if len(said) == 1 else f"{', '.join(said[:-1])} and {said[-1]}"
        return self.template.format(items=joined)

    def _turns(self, run_id: str | None) -> deque[tuple[float, list[Unsure]]] | None:
        return self._runs.get(run_id) if run_id in self._runs else self._runs.get(None)

    def _forget(self, run_id: str | None, used: set[Unsure]) -> None:
        turns = self._turns(run_id)
        if turns is None:
            return
        for index in range(len(turns)):
            at, items = turns[index]
            turns[index] = (at, [item for item in items if item not in used])

    def _scan(self, words: list[Word]) -> list[Unsure]:
        found: list[Unsure] = []
        i = 0
        while i < len(words):
            token = words[i].text.strip(_EDGE)
            low = token.lower()
            if low in NUMBER_WORDS:
                end = i + 1
                while end < len(words) and _number_word(words, end):
                    end += 1
                group = words[i:end]
                value = words_number([w.text.strip(_EDGE) for w in group])
                if value is not None:
                    text = " ".join(w.text.strip(_EDGE) for w in group)
                    self._add(found, text, "number", group, (str(value),))
                i = end
                continue
            if _DIGITS.search(token):
                keys = tuple(_number_key(m[0]) for m in _DIGITS.finditer(token))
                self._add(found, token, "number", [words[i]], keys)
            elif low in _MONTH_OF:
                self._add(found, token, "date", [words[i]], (low, f"-{_MONTH_OF[low]:02d}-"))
            elif low in self.names or (token[:1].isupper() and _mid_sentence(words, i)):
                self._add(found, token, "name", [words[i]], (low,))
            i += 1
        return found

    def _add(
        self, found: list[Unsure], text: str, kind: str, group: list[Word], keys: tuple[str, ...]
    ) -> None:
        scores = [w.confidence for w in group if w.confidence is not None]
        if scores and min(scores) < self.threshold:
            found.append(Unsure(text, kind, min(scores), keys))


def _mid_sentence(words: list[Word], index: int) -> bool:
    return (
        index > 0
        and words[index].text != "I"
        and not words[index - 1].text.endswith((".", "?", "!"))
    )


def _number_word(words: list[Word], index: int) -> bool:
    low = words[index].text.strip(_EDGE).lower()
    if low == "and":
        return index + 1 < len(words) and _number_word(words, index + 1)
    return all(part in NUMBER_WORDS for part in low.split("-"))


def _argument_values(arguments: Mapping[str, Any]) -> tuple[set[str], str]:
    numbers: set[str] = set()
    texts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for inner in value.values():
                walk(inner)
        elif isinstance(value, list | tuple):
            for inner in value:
                walk(inner)
        elif isinstance(value, bool) or value is None:
            return
        elif isinstance(value, int | float):
            numbers.add(_number_key(str(value)))
        else:
            text = str(value)
            texts.append(text.lower())
            numbers.update(_number_key(m[0]) for m in _DIGITS.finditer(text))

    walk(arguments)
    return numbers, " | ".join(texts)


def _shows_in(item: Unsure, numbers: set[str], texts: str) -> bool:
    if item.kind == "number":
        return any(key in numbers for key in item.keys)
    return any(key in texts for key in item.keys)
