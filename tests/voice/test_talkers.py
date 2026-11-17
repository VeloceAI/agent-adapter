from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from agent_adapter.voice import _open_models
from agent_adapter.voice.talker import Cue, Talker, backchannel

from .open_model_fakes import DuplexSession, Loads, fake_haru, fake_moshi

LOUD_24K = b"\x00\x10" * 3_200  # one 133 ms Haru frame of a steady signal
QUIET_24K = bytes(2 * 3_200)


@pytest.fixture(autouse=True)
def fresh_models() -> Iterator[None]:
    _open_models.forget_models()
    Loads.count = 0
    yield
    _open_models.forget_models()


async def _cues(stream: object, *chunks: bytes) -> list[Cue]:
    for chunk in chunks:
        stream.hear(chunk)  # type: ignore[attr-defined]
    await stream.close()  # type: ignore[attr-defined]
    return [cue async for cue in stream.cues()]  # type: ignore[attr-defined]


async def test_haru_answers_speech_with_cues_and_skips_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = fake_haru(monkeypatch)
    from agent_adapter.voice.haru import HaruTalker

    talker = HaruTalker(system_prompt="You are Ana.", voice="ana.wav")
    assert isinstance(talker, Talker)
    await talker.warm()
    cues = await _cues(talker.open(24_000), QUIET_24K, LOUD_24K, QUIET_24K, QUIET_24K)
    assert [cue.text for cue in cues] == ["mm-hm", "", ""]  # speech, then 2 frames of tail
    assert len(cues[0].audio) == 2 * 3_200
    assert DuplexSession.last.options == {
        "system_prompt": "You are Ana.",
        "reference": "ref:wav:ana.wav",
    }
    await _cues(talker.open(24_000), LOUD_24K)
    assert prepared == ["wav:ana.wav"] and Loads.count == 1  # voice and model reused


async def test_haru_resamples_caller_audio_and_ends_when_the_session_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_haru(monkeypatch)
    from agent_adapter.voice.haru import HaruTalker

    stream = HaruTalker().open(16_000)
    cues = await _cues(stream, b"\x00\x10" * (3_200 * 2 // 3 * 6))  # six frames at 16 kHz
    assert len(DuplexSession.last.heard) == DuplexSession.length
    assert "reference" not in DuplexSession.last.options
    assert cues and all(c.text == "mm-hm" for c in cues)


async def test_moshi_talks_back_on_its_own_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_moshi(monkeypatch)
    from agent_adapter.voice.moshi_talker import MoshiTalker

    talker = MoshiTalker(device="cpu", temp=0.6)
    assert isinstance(talker, Talker)
    await talker.warm()
    mimi, _lm, _tok = talker._model.get().model
    assert mimi.codebooks == 8
    cues = await _cues(talker.open(24_000), b"\x00\x10" * 12, bytes(2 * 4 * 4))
    assert [c.text for c in cues] == [" w11", " w12", "", ""]
    assert mimi.streams == 1


async def test_backchannel_plays_cues_only_while_the_agent_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_haru(monkeypatch)
    from agent_adapter.voice.haru import HaruTalker

    stream = HaruTalker().open(24_000)
    played: list[bytes] = []
    agent_speaking = False

    async def play(audio: bytes) -> None:
        nonlocal agent_speaking
        played.append(audio)
        agent_speaking = True  # the agent's answer starts after the first cue

    task = asyncio.create_task(backchannel(stream, play, may_speak=lambda: not agent_speaking))
    stream.hear(LOUD_24K * 3)
    await stream.close()
    await task
    assert len(played) == 1


async def test_backchannel_takes_a_sync_player(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_haru(monkeypatch)
    from agent_adapter.voice.haru import HaruTalker

    stream = HaruTalker().open(24_000)
    played: list[bytes] = []
    stream.hear(LOUD_24K)
    await stream.close()
    await backchannel(stream, played.append, may_speak=lambda: True)
    assert len(played) == 1
