"""Chatterbox text-to-speech from Resemble AI (code and weights MIT; archived in 2026).

Chatterbox renders a whole clip per call, so a line is split at sentence ends and each
sentence plays as soon as it is ready: the caller hears the first sentence while the
rest renders. Every clip carries Resemble's PerTh watermark.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

from agent_adapter.core.errors import MissingDependency
from agent_adapter.voice._open_models import Lazy, flat, pcm16, sentences, thread_stream
from agent_adapter.voice.pcm import Resampler

Variant = Literal["standard", "turbo", "multilingual"]
_PIP = 'pip install "agent-adapter[voice-chatterbox]"'


def _load(variant: Variant, device: str, nano: bool) -> Any:
    try:
        if variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            return ChatterboxTurboTTS.from_pretrained(device=device, nano=nano)
        if variant == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            return ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        from chatterbox.tts import ChatterboxTTS as Standard

        return Standard.from_pretrained(device=device)
    except ImportError as exc:
        raise MissingDependency(f"ChatterboxTTS needs chatterbox-tts: {_PIP}") from exc


def _check_installed() -> None:
    try:
        import chatterbox  # noqa: F401
    except ImportError as exc:
        raise MissingDependency(f"ChatterboxTTS needs chatterbox-tts: {_PIP}") from exc


class ChatterboxTTS:
    """Chatterbox voices at 24 kHz; `voice` is a reference clip (a path) to clone.

    `variant="turbo"` (350M, one-step decoder) is the fastest and needs `voice`;
    `nano=True` picks its 110M model for the CPU. `variant="multilingual"` needs
    `language` (for example "fr"). `exaggeration` and `cfg_weight` apply to the standard
    and multilingual models.
    """

    sample_rate = 24_000

    def __init__(
        self,
        *,
        variant: Variant = "turbo",
        voice: str | None = None,
        language: str | None = None,
        device: str = "cuda",
        nano: bool = False,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> None:
        if variant == "turbo" and voice is None:
            raise ValueError("Chatterbox Turbo needs voice= (a reference clip of about 10 s)")
        if variant == "multilingual" and not language:
            raise ValueError('multilingual Chatterbox needs language= (for example "fr")')
        _check_installed()
        self._options: dict[str, Any] = {}
        if voice is not None:
            self._options["audio_prompt_path"] = voice
        if variant != "turbo":
            self._options.update(exaggeration=exaggeration, cfg_weight=cfg_weight)
        if variant == "multilingual":
            self._options["language_id"] = language
        self._model = Lazy(
            ("chatterbox", variant, device, nano), lambda: _load(variant, device, nano)
        )

    async def warm(self) -> None:
        """Load the model and render one short word, so the first line is fast."""
        async for _ in self.synthesize("Hi."):
            pass

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        warm = await self._model.aget()

        def produce(emit: Callable[[bytes], None], stop: threading.Event) -> None:
            for sentence in sentences(text):
                if stop.is_set():
                    return
                with warm.lock:
                    wav = warm.model.generate(sentence, **self._options)
                rate = int(getattr(warm.model, "sr", self.sample_rate))
                pcm = pcm16(flat(wav))
                if rate != self.sample_rate:
                    pcm = Resampler(rate, self.sample_rate).feed(pcm)
                emit(pcm)

        async for pcm in thread_stream(produce):
            yield pcm


__all__ = ["ChatterboxTTS"]
