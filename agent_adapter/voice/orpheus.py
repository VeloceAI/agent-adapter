"""Orpheus text-to-speech from Canopy Labs (code Apache-2.0; weights Apache-2.0 on a
Llama 3.2 base). It runs on vLLM, so it needs a CUDA GPU on Linux.

Orpheus streams: the first chunk arrives in about 200 ms and plays while the rest renders.
Emotive tags such as `<laugh>` and `<sigh>` in the text are spoken as sounds.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

from agent_adapter.core.errors import MissingDependency
from agent_adapter.voice._open_models import Lazy, thread_stream


def _orpheus() -> Any:
    try:
        import orpheus_tts
    except ImportError as exc:
        raise MissingDependency(
            'OrpheusTTS needs orpheus-speech: pip install "agent-adapter[voice-orpheus]"'
        ) from exc
    return orpheus_tts


class OrpheusTTS:
    """Orpheus voices as PCM16 at 24 kHz; English voices: tara, leah, jess, leo, dan, mia,
    zac, zoe. Extra keyword `options` go to `generate_speech` (for example
    `repetition_penalty=1.1`, which the Orpheus README says must stay at 1.1 or more)."""

    sample_rate = 24_000

    def __init__(
        self,
        model: str = "canopylabs/orpheus-tts-0.1-finetune-prod",
        *,
        voice: str = "tara",
        max_model_len: int = 2048,
        **options: Any,
    ) -> None:
        _orpheus()
        self.voice = voice
        self.options = options

        def load() -> Any:
            return _orpheus().OrpheusModel(model_name=model, max_model_len=max_model_len)

        self._model = Lazy(("orpheus", model, max_model_len), load)

    async def warm(self) -> None:
        """Load the model and render one short word, so the first line is fast."""
        async for _ in self.synthesize("Hi."):
            pass

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        warm = await self._model.aget()

        def produce(emit: Callable[[bytes], None], stop: threading.Event) -> None:
            with warm.lock:
                chunks = warm.model.generate_speech(prompt=text, voice=self.voice, **self.options)
                try:
                    for chunk in chunks:
                        if stop.is_set():
                            return
                        emit(bytes(chunk))
                finally:
                    close = getattr(chunks, "close", None)
                    if close is not None:
                        close()

        async for pcm in thread_stream(produce):
            yield pcm


__all__ = ["OrpheusTTS"]
