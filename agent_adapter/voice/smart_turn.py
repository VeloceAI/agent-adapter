"""End of turn from what the caller said, not just how long they paused: Smart Turn v3.

A pause after "I'd like to book a seat for" is not the end of a turn. `SmartTurn` wraps any
`TurnDetector`; when it hears a pause, the Smart Turn model looks at the turn's audio and the
turn ends only if the model says the caller has finished (or after `max_wait_ms` more quiet).

Needs `pip install "agent-adapter[vad]"` and `smart-turn-v3.*.onnx` from the
pipecat-ai/smart-turn-v3 model page (BSD-2-Clause), or the copy Pipecat ships under
`pipecat/audio/turn/smart_turn/data`.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from agent_adapter.voice.turns import TurnDetector, TurnEvent, pcm16

RATE = 16_000
_SECONDS = 8
_N_FFT, _HOP, _MELS = 400, 160, 80
_PRE_ROLL_MS = 500  # a detector says "started" a little after the first syllable


def _slaney_mel(hz: Any, np: Any) -> Any:
    mels = 3.0 * hz / 200.0
    log_region = hz >= 1000.0
    return np.where(
        log_region, 15.0 + np.log(np.maximum(hz, 1e-9) / 1000.0) * 27.0 / np.log(6.4), mels
    )


def _slaney_hz(mels: Any, np: Any) -> Any:
    hz = 200.0 * mels / 3.0
    return np.where(mels >= 15.0, 1000.0 * np.exp(np.log(6.4) / 27.0 * (mels - 15.0)), hz)


@cache
def _filters(np: Any) -> Any:
    edges = _slaney_hz(np.linspace(0.0, _slaney_mel(np.float64(8000.0), np), _MELS + 2), np)
    bins = np.linspace(0, RATE // 2, _N_FFT // 2 + 1)
    slopes = edges[np.newaxis, :] - bins[:, np.newaxis]
    gaps = np.diff(edges)
    down = -slopes[:, :-2] / gaps[:-1]
    up = slopes[:, 2:] / gaps[1:]
    bank = np.maximum(0.0, np.minimum(down, up))
    return bank * (2.0 / (edges[2:] - edges[:-2]))  # Slaney area normalisation


def log_mel(audio: Any, np: Any) -> Any:
    """Whisper's log-mel features of 8 s of 16 kHz float32 audio: an (80, 800) array."""
    x = np.asarray(audio, dtype=np.float32)
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    padded = np.pad(x.astype(np.float64), _N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, _N_FFT)[::_HOP]
    window = np.hanning(_N_FFT + 1)[:-1]
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    spec = np.log10(np.maximum(1e-10, power @ _filters(np))).T[:, :-1]
    spec = np.maximum(spec, spec.max() - 8.0)
    return ((spec + 4.0) / 4.0).astype(np.float32)


class SmartTurn:
    """A `TurnDetector` for 16 kHz 16-bit mono PCM: `detector` hears pauses, the model
    decides whether a pause ends the turn."""

    def __init__(
        self,
        model_path: str | Path,
        detector: TurnDetector,
        *,
        threshold: float = 0.5,
        max_wait_ms: float = 3_000,
    ) -> None:
        if not 0 < threshold < 1 or max_wait_ms <= 0:
            raise ValueError("0 < threshold < 1 and max_wait_ms > 0")
        import numpy as np
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        self._np: Any = np
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.detector = detector
        self.threshold = threshold
        self.max_wait_ms = max_wait_ms
        self.speaking = False
        self.last_probability: float | None = None
        self._audio = np.zeros(0, dtype=np.float32)
        self._waited_ms: float | None = None  # quiet since a pause the model said was mid-turn
        self._odd = b""

    def probability(self, audio: Any) -> float:
        """How sure the model is that a turn ending with `audio` (float32, 16 kHz) is done."""
        np, size = self._np, RATE * _SECONDS
        audio = np.asarray(audio, dtype=np.float32)[-size:]
        audio = np.pad(audio, (size - len(audio), 0))
        features = log_mel(audio, np)[np.newaxis]
        (out,) = self._session.run(None, {"input_features": features})
        return float(np.asarray(out).reshape(-1)[0])

    def feed(self, pcm: bytes) -> list[TurnEvent]:
        np = self._np
        data = self._odd + pcm
        cut = len(data) - len(data) % 2
        self._odd = data[cut:]
        samples = np.asarray(pcm16(data[:cut]), dtype=np.float32) / 32_768
        self._audio = np.concatenate([self._audio, samples])[-RATE * _SECONDS :]
        out: list[TurnEvent] = []
        for event in self.detector.feed(pcm):
            if event == "started":
                self._waited_ms = None
                if not self.speaking:
                    self.speaking = True
                    self._audio = self._audio[-RATE * _PRE_ROLL_MS // 1000 :]
                    out.append("started")
            elif self.speaking:
                self.last_probability = self.probability(self._audio)
                if self.last_probability > self.threshold:
                    out += self._end()
                else:
                    self._waited_ms = 0.0
        if self._waited_ms is not None and not out:
            self._waited_ms += len(samples) * 1000 / RATE
            if self._waited_ms >= self.max_wait_ms:
                out += self._end()
        return out

    def _end(self) -> list[TurnEvent]:
        self.speaking = False
        self._waited_ms = None
        return ["ended"]


__all__ = ["RATE", "SmartTurn", "log_mel"]
