"""`NoiseGate`: a spectral noise filter for the caller's audio (`VoiceSession(filters=[...])`)."""

from __future__ import annotations

from typing import Any

from agent_adapter.core.errors import MissingDependency
from agent_adapter.voice.pcm import bytes_of, samples_of


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise MissingDependency('NoiseGate needs numpy: pip install "agent-adapter[vad]"') from exc
    return numpy


class NoiseGate:
    """Learns the steady background noise in each frequency band and turns it down by up to
    `reduction_db`, leaving speech alone. 16 kHz PCM16 in and out, same length, about
    `frame / 16` ms later. Steady noise (fans, hum, road) it handles; other voices it cannot.

    Any callable `pcm -> pcm` works as a filter, so a stronger denoiser plugs in the same way.
    """

    def __init__(
        self,
        *,
        reduction_db: float = 12.0,
        frame: int = 512,
        rise: float = 0.995,
        fall: float = 0.9,
        smooth: float = 0.5,
        over: float = 2.5,
    ) -> None:
        np = self._np = _numpy()
        self.over = over
        self._power: Any = None
        self.frame, self.hop = frame, frame // 2
        self.floor = 10 ** (-reduction_db / 20)
        self.rise, self.fall, self.smooth = rise, fall, smooth
        self._window = np.sqrt(np.hanning(frame + 1)[:-1])  # periodic: overlap-add sums to 1
        self._in = np.zeros(frame, dtype=np.float64)
        self._pending = np.zeros(0, dtype=np.float64)
        self._tail = np.zeros(self.hop, dtype=np.float64)
        self._out = np.zeros(frame, dtype=np.float64)  # the filter's delay
        self._noise: Any = None
        self._gain: Any = None

    def __call__(self, pcm: bytes) -> bytes:
        np = self._np
        fresh = np.asarray(samples_of(pcm), dtype=np.float64)
        self._pending = np.concatenate([self._pending, fresh])
        made = []
        while len(self._pending) >= self.hop:
            hop, self._pending = self._pending[: self.hop], self._pending[self.hop :]
            self._in = np.concatenate([self._in[self.hop :], hop])
            made.append(self._step(self._in))
        if made:
            self._out = np.concatenate([self._out, *made])
        out, self._out = self._out[: len(fresh)], self._out[len(fresh) :]
        clipped = np.clip(np.round(out), -32768, 32767).astype(np.int16)
        return bytes_of(_array(clipped))

    def _step(self, frame: Any) -> Any:
        np = self._np
        spectrum = np.fft.rfft(frame * self._window)
        power = np.abs(spectrum) ** 2
        if self._noise is None:
            self._noise, self._power = power.copy(), power.copy()
        self._power = 0.6 * self._power + 0.4 * power  # a steadier view of each band
        rate = np.where(self._power < self._noise, self.fall, self.rise)
        self._noise = rate * self._noise + (1 - rate) * self._power
        # tracking the low points under-reads the noise mean: `over` makes up for it
        snr = np.maximum(self._power / np.maximum(self.over * self._noise, 1e-9) - 1, 0)
        gain = np.maximum(snr / (1 + snr), self.floor)
        if self._gain is not None:
            gain = self.smooth * self._gain + (1 - self.smooth) * gain
        self._gain = gain
        frame_out = np.fft.irfft(spectrum * gain, n=self.frame) * self._window
        out = self._tail + frame_out[: self.hop]
        self._tail = frame_out[self.hop :]
        return out


def _array(values: Any) -> Any:
    from array import array

    return array("h", values.tobytes())


__all__ = ["NoiseGate"]
