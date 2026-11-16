"""16-bit PCM helpers with no dependencies: streaming resampling and G.711 μ-law.

`audioop` left the standard library in Python 3.13, so these are written out here.
"""

from __future__ import annotations

import math
import sys
from array import array
from functools import cache


def samples_of(pcm: bytes) -> array[int]:
    """16-bit little-endian PCM as native integers (a trailing odd byte is dropped)."""
    samples = array("h", pcm[: len(pcm) & ~1])
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def bytes_of(samples: array[int]) -> bytes:
    """Native 16-bit integers as little-endian PCM."""
    if sys.byteorder == "big":
        samples = array("h", samples)
        samples.byteswap()
    return samples.tobytes()


def rms(pcm: bytes) -> float:
    """Root-mean-square level as a fraction of full scale (0 for no audio)."""
    samples = samples_of(pcm)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples)) / 32_768


class Resampler:
    """Changes the sample rate of a PCM16 mono stream, chunk by chunk, with no seams.

    Linear interpolation; when the rate drops by 2x or more, a box filter first removes
    most of what would alias. Good enough for speech models and phone audio.
    """

    def __init__(self, from_rate: int, to_rate: int) -> None:
        if from_rate <= 0 or to_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.from_rate = from_rate
        self.to_rate = to_rate
        self._step = from_rate / to_rate
        self._box = int(self._step) if self._step >= 2 else 1
        self._history: list[int] = []  # the last box-1 inputs, for the filter across chunks
        self._carry: list[int] = []  # inputs the next output still needs
        self._pos = 0.0
        self._odd = b""

    def feed(self, pcm: bytes) -> bytes:
        data, self._odd = self._odd + pcm, b""
        if len(data) % 2:
            data, self._odd = data[:-1], data[-1:]
        if self.from_rate == self.to_rate:
            return data
        source = self._carry + self._filtered(list(samples_of(data)))
        out = array("h")
        pos, step, last = self._pos, self._step, len(source) - 1
        while pos < last:
            i = int(pos)
            a = source[i]
            out.append(round(a + (source[i + 1] - a) * (pos - i)))
            pos += step
        keep = min(int(pos), len(source))
        self._carry, self._pos = source[keep:], pos - keep
        return bytes_of(out)

    def _filtered(self, samples: list[int]) -> list[int]:
        if self._box == 1:
            return samples
        box, joined = self._box, self._history + samples
        total = sum(joined[: box - 1])
        out = []
        for i in range(box - 1, len(joined)):
            total += joined[i]
            out.append(total // box)
            total -= joined[i - box + 1]
        missing = box - 1 - len(self._history)  # the very first samples have no history yet
        self._history = joined[-(box - 1) :]
        return samples[:missing] + out if missing > 0 else out


_BIAS = 0x84
_CLIP = 32_635


def _encode_one(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    magnitude = min(-sample if sign else sample, _CLIP) + _BIAS
    exponent = (magnitude >> 7).bit_length() - 1
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return ~(sign | exponent << 4 | mantissa) & 0xFF


def _decode_one(code: int) -> int:
    code = ~code & 0xFF
    magnitude = (((code & 0x0F) << 3) + _BIAS) << ((code >> 4) & 0x07)
    return _BIAS - magnitude if code & 0x80 else magnitude - _BIAS


@cache
def _encode_table() -> bytes:
    return bytes(_encode_one(s - 65_536 if s >= 32_768 else s) for s in range(65_536))


@cache
def _decode_table() -> array[int]:
    return array("h", (_decode_one(c) for c in range(256)))


def ulaw_encode(pcm: bytes) -> bytes:
    """PCM16 to G.711 μ-law, one byte per sample."""
    table = _encode_table()
    return bytes(table[s & 0xFFFF] for s in samples_of(pcm))


def ulaw_decode(data: bytes) -> bytes:
    """G.711 μ-law to PCM16."""
    table = _decode_table()
    return bytes_of(array("h", (table[c] for c in data)))


__all__ = ["Resampler", "bytes_of", "rms", "samples_of", "ulaw_decode", "ulaw_encode"]
