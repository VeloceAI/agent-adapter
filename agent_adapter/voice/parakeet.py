"""NVIDIA Parakeet speech-to-text through NeMo (weights CC-BY-4.0; NeMo Apache-2.0).

`parakeet-tdt-0.6b-v3` covers 25 European languages; `-v2` is English only. NeMo needs
Python 3.12 or newer and PyTorch; give it its own environment.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_adapter.core.errors import MissingDependency
from agent_adapter.voice._open_models import Lazy, floats
from agent_adapter.voice.pcm import Resampler

_PIP = 'pip install "agent-adapter[voice-parakeet]"'


def _nemo_asr() -> Any:
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise MissingDependency(f"ParakeetSTT needs nemo_toolkit[asr]: {_PIP}") from exc
    return nemo_asr


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise MissingDependency(f"ParakeetSTT needs numpy: {_PIP}") from exc
    return numpy


def _text(hypothesis: Any) -> str:
    return str(getattr(hypothesis, "text", hypothesis)).strip()  # NeMo 1.x returned strings


class ParakeetSTT:
    """Speech-to-text for one finished turn, on a GPU (`device="cuda"`) or the CPU."""

    rate = 16_000

    def __init__(self, model: str = "nvidia/parakeet-tdt-0.6b-v3", *, device: str | None = None):
        _nemo_asr()

        def load() -> Any:
            asr = _nemo_asr().models.ASRModel.from_pretrained(model_name=model)
            if device is not None:
                asr = asr.to(device)
            asr.eval()
            return asr

        self._model = Lazy(("parakeet", model, device), load)

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
                out = warm.model.transcribe([audio], batch_size=1, verbose=False)
            return _text(out[0]) if out else ""

        return await asyncio.to_thread(run)


__all__ = ["ParakeetSTT"]
