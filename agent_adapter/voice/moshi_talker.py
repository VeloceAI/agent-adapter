"""Moshi from Kyutai as a talker (code MIT; weights CC-BY-4.0). Its model card says it is
for research only, so the docs list it as research only.

Moshi listens and speaks at once in 80 ms frames (1920 samples at 24 kHz).
"""

from __future__ import annotations

from typing import Any

from agent_adapter.voice._duplex import DuplexStream, Frames, Out
from agent_adapter.voice._open_models import Lazy
from agent_adapter.voice.kyutai import RATE, _moshi


def _load(repo: str, device: str) -> tuple[Any, Any, Any]:
    models, _torch = _moshi()
    info = models.loaders.CheckpointInfo.from_hf_repo(repo)
    mimi = info.get_mimi(device=device)
    mimi.set_num_codebooks(8)  # Moshi uses 8 of Mimi's codebooks
    return mimi, info.get_moshi(device=device), info.get_text_tokenizer()


class MoshiTalker:
    """A `Talker` on Moshi; `model` is a Moshi repo (`moshiko` male, `moshika` female)."""

    sample_rate = RATE

    def __init__(
        self,
        model: str = "kyutai/moshiko-pytorch-bf16",
        *,
        device: str = "cuda",
        temp: float = 0.8,
        temp_text: float = 0.7,
    ) -> None:
        _moshi()
        self.device = device
        self.temp, self.temp_text = temp, temp_text
        self._model = Lazy(("moshi", model, device), lambda: _load(model, device))

    async def warm(self) -> None:
        """Load the model ahead of the first call."""
        await self._model.aget()

    def open(self, sample_rate: int) -> DuplexStream:
        return DuplexStream(self._run, in_rate=sample_rate, model_rate=self.sample_rate)

    def _run(self, frames: Frames, out: Out) -> None:
        models, torch = _moshi()
        warm = self._model.get()
        mimi, lm, tokenizer = warm.model
        with warm.lock:
            gen = models.LMGen(lm, temp=self.temp, temp_text=self.temp_text)
            with torch.no_grad(), gen.streaming(1), mimi.streaming(1):
                for samples in frames(int(mimi.frame_size)):
                    audio = torch.tensor(samples, dtype=torch.float32, device=self.device)
                    tokens = gen.step(mimi.encode(audio.reshape(1, 1, -1)))
                    if tokens is None:
                        continue
                    token = int(tokens[0, 0, 0].item())
                    text = "" if token in (0, 3) else tokenizer.id_to_piece(token)
                    out(mimi.decode(tokens[:, 1:])[0, 0], text.replace("▁", " "))


__all__ = ["MoshiTalker"]
