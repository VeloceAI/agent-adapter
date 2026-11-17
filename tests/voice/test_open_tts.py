from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator

import pytest

from agent_adapter.voice import _open_models
from agent_adapter.voice.media import TTS

from .open_model_fakes import Loads, OrpheusModel, TTSModel, fake_chatterbox, fake_moshi, module


@pytest.fixture(autouse=True)
def fresh_models() -> Iterator[None]:
    _open_models.forget_models()
    Loads.count = 0
    yield
    _open_models.forget_models()


async def _all(tts: TTS, text: str) -> list[bytes]:
    return [chunk async for chunk in tts.synthesize(text)]


async def _until(check: object, seconds: float = 5.0) -> None:
    for _ in range(int(seconds * 100)):
        if callable(check) and check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


async def test_chatterbox_turbo_plays_each_sentence_as_it_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_chatterbox(monkeypatch)
    from agent_adapter.voice.chatterbox_tts import ChatterboxTTS

    tts = ChatterboxTTS(voice="me.wav", nano=True, device="cpu")
    assert isinstance(tts, TTS)
    chunks = await _all(tts, "Your order shipped. It arrives Friday!")
    assert [len(c) for c in chunks] == [
        2 * len("Your order shipped."),
        2 * len("It arrives Friday!"),
    ]
    model = tts._model.get().model
    assert type(model).__name__ == "ChatterboxTurboTTS"
    assert model.load == {"device": "cpu", "nano": True}
    assert model.calls[0] == ("Your order shipped.", {"audio_prompt_path": "me.wav"})


async def test_chatterbox_multilingual_passes_language_and_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_chatterbox(monkeypatch)
    from agent_adapter.voice.chatterbox_tts import ChatterboxTTS

    tts = ChatterboxTTS(variant="multilingual", language="fr", exaggeration=0.7)
    await tts.warm()
    model = tts._model.get().model
    assert model.load == {"device": "cuda", "t3_model": "v3"}
    assert model.calls[0] == ("Hi.", {"exaggeration": 0.7, "cfg_weight": 0.5, "language_id": "fr"})


async def test_chatterbox_standard_resamples_to_24_khz(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_chatterbox(monkeypatch)
    from agent_adapter.voice.chatterbox_tts import ChatterboxTTS

    tts = ChatterboxTTS(variant="standard")
    tts._model.get().model.sr = 48_000
    assert sum(len(c) for c in await _all(tts, "abcdefgh")) == 8  # 8 samples -> 4


def test_chatterbox_checks_what_each_variant_needs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_chatterbox(monkeypatch)
    from agent_adapter.voice.chatterbox_tts import ChatterboxTTS

    with pytest.raises(ValueError, match="reference clip"):
        ChatterboxTTS()
    with pytest.raises(ValueError, match="language="):
        ChatterboxTTS(variant="multilingual")


async def test_orpheus_streams_pcm_chunks_as_vllm_makes_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module(monkeypatch, "orpheus_tts", OrpheusModel=OrpheusModel)
    from agent_adapter.voice.orpheus import OrpheusTTS

    tts = OrpheusTTS(voice="leo", repetition_penalty=1.1)
    assert await _all(tts, "Hello <laugh>") == [b"\x01\x00" * 4, b"\x02\x00" * 4, b"\x03\x00" * 4]
    model = OrpheusModel.last
    assert model.args == ("canopylabs/orpheus-tts-0.1-finetune-prod", 2048)
    assert model.calls == [{"prompt": "Hello <laugh>", "voice": "leo", "repetition_penalty": 1.1}]


async def test_orpheus_stops_rendering_on_a_barge_in(monkeypatch: pytest.MonkeyPatch) -> None:
    module(monkeypatch, "orpheus_tts", OrpheusModel=OrpheusModel)
    monkeypatch.setattr(OrpheusModel, "gate", threading.Event())
    from agent_adapter.voice.orpheus import OrpheusTTS

    tts = OrpheusTTS()
    stream = tts.synthesize("A long answer.")
    assert await anext(stream) == b"\x01\x00" * 4
    await stream.aclose()  # the caller cut in
    assert OrpheusModel.gate is not None
    OrpheusModel.gate.set()
    await _until(lambda: OrpheusModel.last.closed)


async def test_kyutai_tts_streams_frames_and_prepares_the_voice_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_moshi(monkeypatch)
    from agent_adapter.voice.kyutai import KyutaiTTS

    tts = KyutaiTTS(device="cpu", temp=0.5)
    assert isinstance(tts, TTS)
    chunks = await _all(tts, "two words")
    assert len(chunks) == 2  # the incomplete first frame is skipped
    assert chunks[0] == _open_models.pcm16([0.25] * 4)
    await _all(tts, "again")
    model = TTSModel.last
    assert model.args == {"n_q": 32, "temp": 0.5, "device": "cpu"}
    assert model.voices == ["/voices/expresso/ex03-ex01_happy_001_channel1_334s.wav.safetensors"]
    assert model.mimi.streams == 2


async def test_kyutai_tts_takes_an_embedding_path_and_stops_on_barge_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_moshi(monkeypatch)
    from agent_adapter.voice.kyutai import KyutaiTTS

    tts = KyutaiTTS(voice="mine.safetensors")
    await tts.warm()
    TTSModel.last.gate = gate = threading.Event()
    stream = tts.synthesize("one two three")
    await anext(stream)
    await stream.aclose()
    gate.set()
    assert TTSModel.last.voices == ["mine.safetensors"]
    await _until(lambda: tts._model.get().lock.acquire(blocking=False))
