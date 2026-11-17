"""Kyutai streaming speech models through the `moshi` package (code MIT; weights CC-BY-4.0).

`KyutaiSTT` decodes every 80 ms frame as it arrives, so at the end of a turn only the
model's own delay is left (0.5 s of audio for `stt-1b-en_fr`, 2.5 s for `stt-2.6b-en`, run
faster than real time). `KyutaiTTS` plays each frame as soon as it is decoded. One loaded
model serves one stream at a time; the others wait for it.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import queue
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from agent_adapter.core.errors import MissingDependency
from agent_adapter.voice._open_models import Lazy, flat, floats, pcm16, thread_stream
from agent_adapter.voice.pcm import Resampler

_PIP = 'pip install "agent-adapter[voice-kyutai]"'
RATE = 24_000
_IDLE_S = 30.0


def _moshi() -> Any:
    try:
        import moshi.models
        import torch
    except ImportError as exc:
        raise MissingDependency(f"Kyutai and Moshi models need moshi and torch: {_PIP}") from exc
    return moshi.models, torch


@dataclass
class _Engine:
    mimi: Any
    tokenizer: Any
    gen: Any
    device: str
    prefix_s: float
    delay_s: float
    padding_id: int


def _load_stt(repo: str, device: str) -> _Engine:
    models, torch = _moshi()
    info = models.loaders.CheckpointInfo.from_hf_repo(repo)
    lm = info.get_moshi(device=device, dtype=torch.bfloat16)
    return _Engine(
        mimi=info.get_mimi(device=device),
        tokenizer=info.get_text_tokenizer(),
        gen=models.LMGen(lm, temp=0, temp_text=0.0),
        device=device,
        prefix_s=float(info.stt_config.get("audio_silence_prefix_seconds", 1.0)),
        delay_s=float(info.stt_config.get("audio_delay_seconds", 5.0)),
        padding_id=int(info.raw_config.get("text_padding_token_id", 3)),
    )


def _decode(engine: _Engine, take: Callable[[], bytes | None], cancelled: threading.Event) -> str:
    """Run one turn through the model: silence prefix, the audio, then silence for the delay."""
    torch = _moshi()[1]
    mimi, frame = engine.mimi, int(engine.mimi.frame_size)
    pieces: list[str] = []

    def step(samples: list[float]) -> None:
        audio = torch.tensor(samples, dtype=torch.float32, device=engine.device)
        tokens = engine.gen.step(mimi.encode(audio.reshape(1, 1, -1)))
        if tokens is None:
            return
        token = int(tokens[0, 0, 0].item())
        if token not in (0, engine.padding_id):
            pieces.append(engine.tokenizer.id_to_piece(token).replace("▁", " "))

    silence = [0.0] * frame
    with torch.no_grad(), mimi.streaming(1), engine.gen.streaming(1):
        for _ in range(math.ceil(engine.prefix_s * mimi.frame_rate)):
            step(silence)
        pending: list[float] = []
        while (pcm := take()) is not None and not cancelled.is_set():
            pending += floats(pcm)
            while len(pending) >= frame:
                step(pending[:frame])
                del pending[:frame]
        if cancelled.is_set():
            return ""
        if pending:
            step(pending + [0.0] * (frame - len(pending)))
        for _ in range(math.ceil(engine.delay_s * mimi.frame_rate)):
            step(silence)
    return "".join(pieces).strip()


class KyutaiSTT:
    """Streaming speech-to-text; `model` is a Kyutai STT repo in the moshi format."""

    def __init__(self, model: str = "kyutai/stt-1b-en_fr", *, device: str = "cuda") -> None:
        _moshi()
        self._model = Lazy(("kyutai-stt", model, device), lambda: _load_stt(model, device))

    async def warm(self) -> None:
        """Load the model and decode half a second of silence, so the first turn is fast."""
        await self.transcribe(bytes(RATE), RATE)

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        stream = self.open(sample_rate)
        stream.feed(pcm)
        return await stream.finish()

    def open(self, sample_rate: int) -> _KyutaiStream:
        return _KyutaiStream(self._model, sample_rate)


class _KyutaiStream:
    def __init__(self, model: Lazy, sample_rate: int) -> None:
        self._audio: queue.Queue[bytes | None] = queue.Queue()
        self._resample = Resampler(sample_rate, RATE) if sample_rate != RATE else None
        self._cancelled = threading.Event()
        self._task = asyncio.ensure_future(asyncio.to_thread(self._run, model))

    def _run(self, model: Lazy) -> str:
        warm = model.get()
        with warm.lock:
            if self._cancelled.is_set():
                return ""
            return _decode(warm.model, self._take, self._cancelled)

    def _take(self) -> bytes | None:
        try:
            return self._audio.get(timeout=_IDLE_S)
        except queue.Empty:  # an abandoned stream must not hold the model forever
            return None

    def feed(self, pcm: bytes) -> None:
        self._audio.put(self._resample.feed(pcm) if self._resample else pcm)

    async def finish(self) -> str:
        self._audio.put(None)
        return await self._task

    async def cancel(self) -> None:
        self._cancelled.set()
        self._audio.put(None)
        with contextlib.suppress(Exception):
            await self._task


def _load_tts(repo: str, device: str, n_q: int, temp: float) -> Any:
    _moshi()
    try:
        from moshi.models.loaders import CheckpointInfo
        from moshi.models.tts import TTSModel
    except ImportError as exc:
        raise MissingDependency(f"KyutaiTTS needs moshi>=0.2.11: {_PIP}") from exc
    info = CheckpointInfo.from_hf_repo(repo)
    return TTSModel.from_checkpoint_info(info, n_q=n_q, temp=temp, device=device)


class _Stopped(Exception):
    pass


class KyutaiTTS:
    """Streaming text-to-speech at 24 kHz; `voice` is a name in `kyutai/tts-voices` or a
    `.safetensors` voice embedding. The voice is prepared once and reused."""

    sample_rate = RATE

    def __init__(
        self,
        model: str = "kyutai/tts-1.6b-en_fr",
        *,
        voice: str = "expresso/ex03-ex01_happy_001_channel1_334s.wav",
        device: str = "cuda",
        n_q: int = 32,
        temp: float = 0.6,
        cfg_coef: float = 2.0,
    ) -> None:
        _moshi()
        self.voice = voice
        self.cfg_coef = cfg_coef
        self._condition: Any = None
        self._model = Lazy(
            ("kyutai-tts", model, device, n_q, temp), lambda: _load_tts(model, device, n_q, temp)
        )

    async def warm(self) -> None:
        """Load the model and render one short word, so the first line is fast."""
        async for _ in self.synthesize("Hi."):
            pass

    def _voice(self, tts: Any) -> Any:
        if self._condition is None:
            path = self.voice
            if not path.endswith(".safetensors"):
                path = tts.get_voice_path(path)
            self._condition = tts.make_condition_attributes([path], cfg_coef=self.cfg_coef)
        return self._condition

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        warm = await self._model.aget()
        torch = _moshi()[1]

        def produce(emit: Callable[[bytes], None], stop: threading.Event) -> None:
            tts = warm.model
            rate = int(tts.mimi.sample_rate)
            resample = Resampler(rate, RATE) if rate != RATE else None

            def on_frame(frame: Any) -> None:
                if stop.is_set():
                    raise _Stopped
                if (frame != -1).all():
                    pcm = pcm16(flat(tts.mimi.decode(frame[:, 1:, :])[0, 0]))
                    emit(resample.feed(pcm) if resample else pcm)

            with warm.lock:
                entries = tts.prepare_script([text], padding_between=1)
                condition = self._voice(tts)
                with contextlib.suppress(_Stopped), torch.no_grad(), tts.mimi.streaming(1):
                    tts.generate([entries], [condition], on_frame=on_frame)

        async for pcm in thread_stream(produce):
            yield pcm


__all__ = ["KyutaiSTT", "KyutaiTTS"]
