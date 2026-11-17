"""`check_stt`: fix each transcript (known names) and note unsure words, around any STT."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from agent_adapter.voice.media import STT, StreamingSTT, STTStream
from agent_adapter.voice.observers import CallObserver
from agent_adapter.voice.quality._unsure import UnsureWords

if TYPE_CHECKING:
    from agent_adapter.voice.session import VoiceSession

Fix = Callable[[str], str]


class CheckedSTT(CallObserver):
    """An STT that runs `fixes` (such as `KnownNames`) on every transcript and gives the words
    to `unsure`. Make one with `check_stt`, and pass it as an observer too, so the words land
    under the call's run id: `VoiceSession(stt=checked, observers=[checked])`.

    Word confidence comes from the STT, or its stream, setting `last_words` (a list of `Word`
    or provider word dicts) with each transcript; without it every word counts as sure,
    though a name the fixes guessed is still read back.
    """

    def __init__(
        self, stt: STT, *, fixes: Sequence[Fix] = (), unsure: UnsureWords | None = None
    ) -> None:
        self.stt = stt
        self.fixes = tuple(fixes)
        self.unsure = unsure
        self.run_id: str | None = None

    def started(self, session: VoiceSession) -> None:
        agent = session.agent
        self.run_id = agent.ctx.run_id if agent is not None else None

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        text = await self.stt.transcribe(pcm, sample_rate)
        return self.check(text, getattr(self.stt, "last_words", None))

    def check(self, text: str, words: Any = None) -> str:
        """Run the fixes on `text` and note its words; returns the fixed text."""
        guessed: list[tuple[str, float]] = []
        for fix in self.fixes:
            text = fix(text)
            guessed += [(f.name, f.score) for f in getattr(fix, "last_fixes", ())]
        if self.unsure is not None and text.strip():
            self.unsure.hear(words or text, run_id=self.run_id, guessed=guessed)
        return text


class CheckedStreamingSTT(CheckedSTT):
    """A `CheckedSTT` around a `StreamingSTT`; it streams too."""

    def open(self, sample_rate: int) -> STTStream:
        stt: Any = self.stt
        return _CheckedStream(stt.open(sample_rate), self)


class _CheckedStream:
    def __init__(self, stream: STTStream, owner: CheckedSTT) -> None:
        self.stream = stream
        self.owner = owner

    def feed(self, pcm: bytes) -> None:
        self.stream.feed(pcm)

    async def finish(self) -> str:
        text = await self.stream.finish()
        words = getattr(self.stream, "last_words", None) or getattr(
            self.owner.stt, "last_words", None
        )
        return self.owner.check(text, words)

    async def cancel(self) -> None:
        await self.stream.cancel()


def check_stt(
    stt: STT, *, fixes: Sequence[Fix] = (), unsure: UnsureWords | None = None
) -> CheckedSTT:
    """Wrap `stt` so its transcripts get `fixes` and `unsure` sees its words; a streaming
    STT stays streaming."""
    kind = CheckedStreamingSTT if isinstance(stt, StreamingSTT) else CheckedSTT
    return kind(stt, fixes=fixes, unsure=unsure)
