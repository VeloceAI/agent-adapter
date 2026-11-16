import math
from array import array

import pytest

from agent_adapter.voice.pcm import (
    Resampler,
    bytes_of,
    rms,
    samples_of,
    ulaw_decode,
    ulaw_encode,
)


def sine(rate: int, hz: float, seconds: float, level: int = 10_000) -> bytes:
    n = int(rate * seconds)
    wave = (int(level * math.sin(2 * math.pi * hz * i / rate)) for i in range(n))
    return bytes_of(array("h", wave))


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (48_000, 16_000),
        (8_000, 16_000),
        (24_000, 48_000),
        (16_000, 8_000),
        (24_000, 16_000),
        (44_100, 16_000),
    ],
)
def test_resampled_length_matches_the_rate_ratio(src, dst):
    out = Resampler(src, dst).feed(sine(src, 300, 1.0))
    assert abs(len(out) // 2 - dst) <= 3


def test_chunked_resampling_is_seamless():
    audio = sine(48_000, 440, 0.5)
    whole = Resampler(48_000, 16_000).feed(audio)
    chunked = Resampler(48_000, 16_000)
    parts = b"".join(chunked.feed(audio[i : i + 962]) for i in range(0, len(audio), 962))
    assert abs(len(parts) - len(whole)) <= 4
    a, b = samples_of(whole), samples_of(parts)
    assert max(abs(x - y) for x, y in zip(a, b, strict=False)) <= 2


def test_upsampling_keeps_the_tone():
    out = samples_of(Resampler(8_000, 16_000).feed(sine(8_000, 200, 0.2)))
    expected = [int(10_000 * math.sin(2 * math.pi * 200 * i / 16_000)) for i in range(len(out))]
    assert max(abs(x - y) for x, y in zip(out, expected, strict=True)) < 300


def test_downsampling_damps_what_would_alias():
    high = Resampler(48_000, 16_000).feed(sine(48_000, 15_000, 0.2))
    low = Resampler(48_000, 16_000).feed(sine(48_000, 300, 0.2))
    assert rms(high) < rms(low) / 3


def test_same_rate_passes_through_and_odd_bytes_carry():
    r = Resampler(16_000, 16_000)
    assert r.feed(b"\x01\x02\x03") == b"\x01\x02"
    assert r.feed(b"\x04") == b"\x03\x04"


def test_bad_rates_are_refused():
    with pytest.raises(ValueError):
        Resampler(0, 16_000)


def test_rms():
    assert rms(b"") == 0
    assert rms(bytes_of(array("h", [16_384, -16_384]))) == pytest.approx(0.5)


def test_ulaw_known_codes():
    assert ulaw_encode(bytes_of(array("h", [0]))) == b"\xff"
    assert samples_of(ulaw_decode(b"\xff\x7f\x00\x80")).tolist() == [0, 0, -32_124, 32_124]


def test_ulaw_round_trip_is_close():
    audio = sine(8_000, 300, 0.1)
    back = samples_of(ulaw_decode(ulaw_encode(audio)))
    for a, b in zip(samples_of(audio), back, strict=True):
        assert abs(a - b) <= max(16, abs(a) // 16)
