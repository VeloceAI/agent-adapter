"""Whisper speech-to-text on CTranslate2 (`faster-whisper`, MIT; Whisper weights MIT).

The model loads once per process and is shared by every `FasterWhisperSTT` with the same
settings. Greedy decoding (`beam_size=1`) and a fixed `language` are the fast defaults for
short spoken turns; raise `beam_size` for harder audio.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_adapter.core.errors import MissingDependency
from agent_adapter.voice._open_models import Lazy, floats
from agent_adapter.voice.pcm import Resampler

_PIP = 'pip install "agent-adapter[voice-whisper]"'


def _faster_whisper() -> Any:
    try:
        import faster_whisper
    except ImportError as exc:
        raise MissingDependency(f"FasterWhisperSTT needs faster-whisper: {_PIP}") from exc
    return faster_whisper


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise MissingDependency(f"FasterWhisperSTT needs numpy: {_PIP}") from exc
    return numpy


class FasterWhisperSTT:
    """Speech-to-text for one finished turn; `model` is a size name, a path or a Hub id.

    Extra keyword `options` go to `WhisperModel.transcribe` as they are (for example
    `vad_filter=True` or `hotwords="Acme"`).
    """

    rate = 16_000

    def __init__(
        self,
        model: str = "small",
        *,
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = "en",
        beam_size: int = 1,
        **options: Any,
    ) -> None:
        _faster_whisper()
        self.language = language
        self.beam_size = beam_size
        self.options = {"condition_on_previous_text": False, **options}

        def load() -> Any:
            return _faster_whisper().WhisperModel(model, device=device, compute_type=compute_type)

        self._model = Lazy(("faster-whisper", model, device, compute_type), load)

    async def warm(self) -> None:
        """Load the model and decode half a second of silence, so the first turn is fast."""
        await self.transcribe(bytes(self.rate), self.rate)

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        warm = await self._model.aget()
        if sample_rate != self.rate:
            pcm = Resampler(sample_rate, self.rate).feed(pcm)
        audio = _numpy().asarray(floats(pcm), dtype="float32")

        def run() -> str:
            with warm.lock:
                segments, _info = warm.model.transcribe(
                    audio, language=self.language, beam_size=self.beam_size, **self.options
                )
                return " ".join(s.text.strip() for s in segments).strip()  # a lazy generator

        return await asyncio.to_thread(run)


__all__ = ["FasterWhisperSTT"]
