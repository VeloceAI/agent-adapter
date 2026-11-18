"""`AnswerCache`: replies to questions heard before, with their audio, so they play at once."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from agent_adapter.voice._speech import sentences
from agent_adapter.voice.media import TTS
from agent_adapter.voice.speculate import same_words

_RECENT_LINES = 32  # audio of lines spoken lately, in case their answer is stored next


@dataclass(frozen=True)
class Answer:
    text: str
    lines: tuple[str, ...]
    expires: float


class AnswerCache:
    """Word-for-word repeats of a question get the same reply, audio included, for `ttl_s`.

    Keys are the tenant plus the question's words (case and punctuation ignored), so two
    tenants never share an entry. A `VoiceAgent` with `Speed(answers=..., tenant=...)` stores
    a reply only when it needed no tool call and opened the call (it depends on the question
    and the instructions alone); `put` seeds known answers. Questions under `min_words` words
    ("yes", "the second one") depend on the call and are never stored.

    Wrap the session's TTS with `voice(tts)`: it keeps the audio of each stored answer and
    plays it back with no synthesis. `VoiceAgent.warm_up` renders seeded answers' audio.
    """

    def __init__(
        self,
        *,
        ttl_s: float = 300.0,
        max_entries: int = 256,
        min_words: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_s <= 0 or max_entries < 1 or min_words < 1:
            raise ValueError("ttl_s > 0, max_entries >= 1 and min_words >= 1")
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.min_words = min_words
        self._clock = clock
        self._entries: OrderedDict[tuple[str, str], Answer] = OrderedDict()
        self._audio: dict[str, list[bytes]] = {}
        self._recent: OrderedDict[str, list[bytes]] = OrderedDict()
        self.hits = 0

    def __len__(self) -> int:
        self._expire()
        return len(self._entries)

    def get(self, tenant: str, question: str) -> Answer | None:
        """The stored reply to `question` for `tenant`, or None."""
        self._expire()
        answer = self._entries.get(_key(tenant, question))
        if answer is not None:
            self.hits += 1
        return answer

    def has(self, tenant: str, question: str) -> bool:
        """Whether `question` has a live answer for `tenant` (not counted as a hit)."""
        self._expire()
        return _key(tenant, question) in self._entries

    def put(self, tenant: str, question: str, reply: str) -> Answer | None:
        """Store `reply` for `question`; None when the question is too short to store."""
        words = same_words(question)
        lines = tuple(sentences(reply))
        if len(words.split()) < self.min_words or not lines:
            return None
        answer = Answer(reply.strip(), lines, self._clock() + self.ttl_s)
        key = _key(tenant, question)
        self._entries.pop(key, None)
        self._entries[key] = answer
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        for line in lines:
            audio = self._recent.pop(line, None)
            if audio is not None:
                self._audio[line] = audio
        return answer

    def forget(self, tenant: str | None = None) -> None:
        """Drop every entry, or one tenant's."""
        for key in [k for k in self._entries if tenant is None or k[0] == tenant]:
            del self._entries[key]
        self._prune_audio()

    def lines(self) -> list[str]:
        """Lines of live answers with no audio yet (rendered at call start)."""
        self._expire()
        live = {line for answer in self._entries.values() for line in answer.lines}
        return sorted(line for line in live if line not in self._audio)

    def audio(self, line: str) -> list[bytes] | None:
        return self._audio.get(line)

    def voice(self, tts: TTS) -> TTS:
        """`tts` that replays stored answers' audio and keeps the audio of new ones."""
        return _CachedVoice(self, tts)

    def _heard(self, line: str, audio: list[bytes]) -> None:
        live = any(line in answer.lines for answer in self._entries.values())
        if live:
            self._audio[line] = audio
            return
        self._recent[line] = audio
        while len(self._recent) > _RECENT_LINES:
            self._recent.popitem(last=False)

    def _expire(self) -> None:
        now = self._clock()
        stale = [key for key, answer in self._entries.items() if answer.expires <= now]
        for key in stale:
            del self._entries[key]
        if stale:
            self._prune_audio()

    def _prune_audio(self) -> None:
        live = {line for answer in self._entries.values() for line in answer.lines}
        for line in [line for line in self._audio if line not in live]:
            del self._audio[line]


class _CachedVoice:
    def __init__(self, cache: AnswerCache, tts: TTS) -> None:
        self._cache = cache
        self._tts = tts
        self.sample_rate = tts.sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        line = text.strip()
        stored = self._cache.audio(line)
        if stored is not None:
            for chunk in stored:
                yield chunk
            return
        audio: list[bytes] = []
        async for chunk in self._tts.synthesize(text):
            audio.append(chunk)
            yield chunk
        if audio:
            self._cache._heard(line, audio)


def _key(tenant: str, question: str) -> tuple[str, str]:
    if not tenant:
        raise ValueError("an answer cache entry needs a tenant")
    return tenant, same_words(question)


__all__ = ["Answer", "AnswerCache"]
