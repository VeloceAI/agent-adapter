import math
from array import array

import pytest

from agent_adapter.voice.turns import SilenceTurns

RATE = 16_000


def tone(ms: float, level: float) -> bytes:
    count = int(RATE * ms / 1000)
    wave = (int(level * 32_767 * math.sin(2 * math.pi * 220 * i / RATE)) for i in range(count))
    return array("h", wave).tobytes()


def run(turns: SilenceTurns, frames: list[bytes]) -> list[str]:
    return [event for frame in frames for event in turns.feed(frame)]


def test_speech_starts_and_ends_a_turn():
    turns = SilenceTurns()
    frames = [tone(20, 0)] * 5 + [tone(20, 0.3)] * 10 + [tone(20, 0)] * 40
    assert run(turns, frames) == ["started", "ended"]
    assert not turns.speaking


def test_a_click_or_a_short_pause_changes_nothing():
    turns = SilenceTurns(start_ms=60, end_ms=600)
    assert run(turns, [tone(20, 0.5), tone(20, 0)] * 5) == []  # 20 ms clicks
    events = run(turns, [tone(20, 0.3)] * 5 + [tone(20, 0)] * 10 + [tone(20, 0.3)] * 5)
    assert events == ["started"]  # a 200 ms pause does not end the turn


def test_frames_of_odd_length_are_joined():
    turns = SilenceTurns(start_ms=0)
    loud = tone(20, 0.3)
    assert turns.feed(loud[:1]) == []
    assert turns.feed(loud[1:]) == ["started"]


def test_bad_settings_are_refused():
    for bad in ({"threshold": 0}, {"threshold": 1.5}, {"end_ms": 0}, {"sample_rate": 0}):
        with pytest.raises(ValueError):
            SilenceTurns(**bad)
