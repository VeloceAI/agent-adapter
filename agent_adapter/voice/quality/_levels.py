"""One frame of PCM16 measured in pure Python: level, peak, clipping, jumps and zero runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from operator import mul, sub

from agent_adapter.voice.pcm import samples_of

FULL_SCALE = 32_768
SILENT_DB = -120.0


def db(fraction: float) -> float:
    """A level as a fraction of full scale, in dBFS (`SILENT_DB` for none)."""
    return 20 * math.log10(fraction) if fraction > 1e-6 else SILENT_DB


@dataclass(frozen=True)
class FrameStats:
    """What one frame of audio looks like."""

    samples: int
    level_db: float  # RMS level, dBFS
    peak: float  # largest sample, fraction of full scale
    clipped: int  # samples at or near full scale
    jump: float  # largest step between two samples, fraction of full scale
    zero_run: int  # longest run of exact zero samples


def measure(pcm: bytes, *, clip_level: int = 32_000, previous: int | None = None) -> FrameStats:
    """Measure one frame; `previous` is the last sample before it, so a click at the seam counts."""
    samples = samples_of(pcm)
    count = len(samples)
    if not count:
        return FrameStats(0, SILENT_DB, 0.0, 0, 0.0, 0)
    high, low = max(samples), min(samples)
    peak = max(high, -low)
    energy = sum(map(mul, samples, samples))
    clipped = 0
    if peak >= clip_level:
        clipped = sum(1 for s in samples if s >= clip_level or s <= -clip_level)
    steps = map(abs, map(sub, samples[1:], samples[:-1]))
    jump = max(steps, default=0)
    if previous is not None:
        jump = max(jump, abs(samples[0] - previous))
    return FrameStats(
        samples=count,
        level_db=db(math.sqrt(energy / count) / FULL_SCALE),
        peak=peak / FULL_SCALE,
        clipped=clipped,
        jump=jump / FULL_SCALE,
        zero_run=_longest_zero_run(pcm[: count * 2]),
    )


def _longest_zero_run(pcm: bytes) -> int:
    """Longest run of zero samples; byte searches keep it in C."""
    best, start, size = 0, 0, len(pcm)
    while start < size:
        found = pcm.find(b"\x00\x00", start)
        if found == -1:
            break
        if found % 2:  # straddles two samples
            start = found + 1
            continue
        rest = pcm[found:]
        run = (len(rest) - len(rest.lstrip(b"\x00"))) // 2
        best = max(best, run)
        start = found + run * 2 + 2
    return best
