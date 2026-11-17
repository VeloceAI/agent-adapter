import math
import struct

import pytest

np = pytest.importorskip("numpy")

from agent_adapter.voice.denoise import NoiseGate  # noqa: E402
from agent_adapter.voice.pcm import rms  # noqa: E402


def pcm(values) -> bytes:
    clipped = np.clip(np.round(values), -32768, 32767).astype("<i2")
    return clipped.tobytes()


def test_steady_noise_is_turned_down_and_length_is_kept():
    rng = np.random.default_rng(1)
    noise = pcm(rng.normal(0, 2000, 16_000 * 2))
    gate = NoiseGate()
    out = b"".join(gate(noise[i : i + 640]) for i in range(0, len(noise), 640))
    assert len(out) == len(noise)
    tail_in, tail_out = noise[-16_000:], out[-16_000:]
    assert rms(tail_out) < rms(tail_in) / 2  # about 12 dB down once it has learnt the noise


def test_a_loud_tone_over_noise_passes():
    rng = np.random.default_rng(2)
    t = np.arange(16_000 * 2) / 16_000
    noise = rng.normal(0, 800, len(t))
    gate = NoiseGate()
    gate(pcm(noise[:16_000]))  # learn the room
    voice = 8000 * np.sin(2 * math.pi * 440 * t[16_000:]) + noise[16_000:]
    out = gate(pcm(voice))
    level = struct.unpack(f"<{len(out) // 2}h", out)
    assert max(abs(s) for s in level[4000:]) > 6000


def test_odd_chunk_sizes_stream_without_seams():
    rng = np.random.default_rng(3)
    audio = pcm(rng.normal(0, 1000, 8000))
    a, b = NoiseGate(), NoiseGate()
    whole = a(audio)
    parts = b"".join(b(audio[i : i + 222]) for i in range(0, len(audio), 222))
    assert whole == parts
