"""Haru-4B from higurashi as a talker (code and weights Apache-2.0; a research release).

Haru listens and speaks at once in 133 ms frames (3200 samples at 24 kHz). It is a
pretrained base model: use it for backchannels and short acknowledgements next to the
agent, never for facts. Clone a voice only with the right to use it.
"""

from __future__ import annotations

from typing import Any

from agent_adapter.core.errors import MissingDependency
from agent_adapter.voice._duplex import DuplexStream, Frames, Out
from agent_adapter.voice._open_models import Lazy

_INSTALL = "Haru is not on PyPI: clone github.com/higurashi-ai/Haru, then pip install -e ./haru"


def _haru() -> Any:
    try:
        import haru
        import haru.audio
        import numpy
    except ImportError as exc:
        raise MissingDependency(f"HaruTalker needs haru and numpy. {_INSTALL}") from exc
    return haru, numpy


class HaruTalker:
    """A `Talker` on Haru-4B. `system_prompt` is Haru's character card (`haru.prompt.render_card`);
    `voice` is a reference clip to speak in, prepared once."""

    sample_rate = 24_000
    frame_samples = 3_200

    def __init__(
        self,
        model: str = "higurashi-ai/Haru-4B",
        *,
        system_prompt: str = "",
        voice: str | None = None,
    ) -> None:
        _haru()
        self.system_prompt = system_prompt
        self.voice = voice
        self._reference: Any = None
        self._model = Lazy(("haru", model), lambda: _haru()[0].HaruModel.from_pretrained(model))

    async def warm(self) -> None:
        """Load the model (and the voice) ahead of the first call."""
        warm = await self._model.aget()
        with warm.lock:
            self._options()

    def open(self, sample_rate: int) -> DuplexStream:
        return DuplexStream(self._run, in_rate=sample_rate, model_rate=self.sample_rate)

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"system_prompt": self.system_prompt}
        if self.voice is not None:
            if self._reference is None:
                audio = _haru()[0].audio
                self._reference = audio.prepare_reference(audio.load_audio(self.voice))
            options["reference"] = self._reference
        return options

    def _run(self, frames: Frames, out: Out) -> None:
        haru, numpy = _haru()
        warm = self._model.get()
        with warm.lock:
            session = haru.DuplexSession(warm.model, **self._options())
            frame = session.start()
            out(frame.audio, frame.text or "")
            for samples in frames(self.frame_samples):
                if frame.done:
                    return
                frame = session.step(numpy.asarray(samples, dtype="float32"))
                out(frame.audio, frame.text or "")


__all__ = ["HaruTalker"]
