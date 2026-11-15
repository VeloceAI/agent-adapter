"""Turns from Silero VAD, run locally with ONNX Runtime: speech, not just loudness.

Needs `pip install "agent-adapter[vad]"` and the model file: `silero_vad.onnx` from the
snakers4/silero-vad release (MIT), or the copy Pipecat ships under `pipecat/audio/vad/data`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_adapter.voice.turns import TurnEvent, TurnTimer, pcm16

_WINDOWS = {16_000: (512, 64), 8_000: (256, 32)}  # samples per window, context carried over
_RESET_AFTER_MS = 2_000  # the model drifts on long silence; Silero's own reference resets too


class SileroTurns:
    """A `TurnDetector` for 16-bit mono PCM at 16 or 8 kHz.

    A window with speech probability >= `threshold` counts as speech; once speaking, it takes
    a drop below `threshold - 0.15` to count as quiet, so trailing syllables don't end a turn.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        sample_rate: int = 16_000,
        threshold: float = 0.5,
        start_ms: float = 60,
        end_ms: float = 600,
    ) -> None:
        if sample_rate not in _WINDOWS:
            raise ValueError("Silero VAD takes 16000 or 8000 Hz audio")
        if not 0.15 < threshold < 1:
            raise ValueError("0.15 < threshold < 1")
        import numpy as np
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        self._np: Any = np
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.timer = TurnTimer(start_ms=start_ms, end_ms=end_ms)
        self._window, self._context_size = _WINDOWS[sample_rate]
        self._window_ms = self._window * 1000 / sample_rate
        self._rate = np.array(sample_rate, dtype=np.int64)
        self._pending = np.zeros(0, dtype=np.float32)
        self._odd = b""
        self._reset_done = False
        self.reset()

    @property
    def speaking(self) -> bool:
        return self.timer.speaking

    def reset(self) -> None:
        """Forget the model's memory of earlier audio."""
        self._state = self._np.zeros((2, 1, 128), dtype=self._np.float32)
        self._context = self._np.zeros(self._context_size, dtype=self._np.float32)

    def probability(self, window: Any) -> float:
        """Speech probability of one window of float32 samples in [-1, 1]."""
        np = self._np
        frame = np.concatenate([self._context, window])[np.newaxis, :]
        out, self._state = self._session.run(
            None, {"input": frame, "state": self._state, "sr": self._rate}
        )
        self._context = frame[0, -self._context_size :]
        return float(out.reshape(-1)[0])

    def feed(self, pcm: bytes) -> list[TurnEvent]:
        np = self._np
        data = self._odd + pcm
        cut = len(data) - len(data) % 2
        self._odd = data[cut:]
        samples = np.asarray(pcm16(data[:cut]), dtype=np.float32) / 32_768
        self._pending = np.concatenate([self._pending, samples])
        events: list[TurnEvent] = []
        while len(self._pending) >= self._window:
            window, self._pending = self._pending[: self._window], self._pending[self._window :]
            floor = self.threshold - 0.15 if self.timer.speaking else self.threshold
            events += self.timer.step(self.probability(window) >= floor, self._window_ms)
            self._maybe_reset()
        return events

    def _maybe_reset(self) -> None:
        if self.timer.speaking or self.timer.quiet_ms < _RESET_AFTER_MS:
            self._reset_done = False
        elif not self._reset_done:
            self.reset()
            self._reset_done = True


__all__ = ["SileroTurns"]
