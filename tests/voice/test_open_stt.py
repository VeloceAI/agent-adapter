from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

from agent_adapter.errors import MissingDependency
from agent_adapter.voice import _open_models
from agent_adapter.voice.media import STT, StreamingSTT

from .open_model_fakes import (
    ASRModel,
    CheckpointInfo,
    Loads,
    WhisperModel,
    fake_moshi,
    fake_nemo,
    fake_numpy,
    module,
)

TONE = b"\x00\x10" * 160  # 10 ms of a constant signal at 16 kHz


@pytest.fixture(autouse=True)
def fresh_models() -> Iterator[None]:
    _open_models.forget_models()
    Loads.count = 0
    yield
    _open_models.forget_models()


@pytest.fixture
def whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_numpy(monkeypatch)
    module(monkeypatch, "faster_whisper", WhisperModel=WhisperModel)


async def test_faster_whisper_transcribes_a_turn_as_float32(whisper: None) -> None:
    from agent_adapter.voice.whisper import FasterWhisperSTT

    stt = FasterWhisperSTT("large-v3-turbo", device="cuda", compute_type="float16")
    assert isinstance(stt, STT)
    assert await stt.transcribe(TONE, 16_000) == "160 samples float32"
    model = stt._model.get().model
    assert model.args == ("large-v3-turbo", "cuda", "float16")
    assert model.calls == [{"language": "en", "beam_size": 1, "condition_on_previous_text": False}]


async def test_faster_whisper_resamples_and_passes_options(whisper: None) -> None:
    from agent_adapter.voice.whisper import FasterWhisperSTT

    stt = FasterWhisperSTT(language=None, beam_size=5, vad_filter=True)
    assert (await stt.transcribe(TONE, 8_000)).startswith("318 samples")  # 160 at 8 kHz
    assert stt._model.get().model.calls[0] == {
        "language": None,
        "beam_size": 5,
        "condition_on_previous_text": False,
        "vad_filter": True,
    }


async def test_one_warm_model_is_shared_by_adapters_with_the_same_settings(whisper: None) -> None:
    from agent_adapter.voice.whisper import FasterWhisperSTT

    first, second, other = FasterWhisperSTT(), FasterWhisperSTT(), FasterWhisperSTT("base")
    assert Loads.count == 0  # nothing loads until it is used
    await first.warm()
    await second.transcribe(TONE, 16_000)
    assert Loads.count == 1
    await other.warm()
    assert Loads.count == 2


async def test_parakeet_transcribes_on_the_chosen_device(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_numpy(monkeypatch)
    fake_nemo(monkeypatch)
    from agent_adapter.voice.parakeet import ParakeetSTT

    stt = ParakeetSTT("nvidia/parakeet-tdt-0.6b-v2", device="cuda")
    await stt.warm()
    assert await stt.transcribe(TONE, 16_000) == "160 samples"
    model: ASRModel = stt._model.get().model
    assert (model.name, model.device, model.evaluated) == (
        "nvidia/parakeet-tdt-0.6b-v2",
        "cuda",
        True,
    )
    assert model.calls[-1] == (1, False)
    model.legacy = True  # NeMo 1.x returned plain strings
    assert await stt.transcribe(TONE, 48_000) == "53 samples"


async def test_kyutai_streams_frames_while_the_caller_talks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_moshi(monkeypatch)
    from agent_adapter.voice.kyutai import KyutaiSTT

    stt = KyutaiSTT(device="cpu")
    assert isinstance(stt, StreamingSTT)
    stream = stt.open(24_000)
    stream.feed(b"\x00\x10" * 6)  # one full frame of 4 samples, and 2 left over
    stream.feed(b"\x00\x10" * 2)
    assert await stream.finish() == "w11 w12"
    engine = stt._model.get().model
    assert engine.gen.temps == (0, 0.0)
    assert engine.gen.steps == 1 + 2 + 2  # prefix silence, the audio, the delay
    assert engine.mimi.streams == 1
    assert CheckpointInfo.last.lm_args == {"device": "cpu", "dtype": "bfloat16"}


async def test_kyutai_transcribe_resamples_a_finished_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_moshi(monkeypatch)
    from agent_adapter.voice.kyutai import KyutaiSTT

    stt = KyutaiSTT()
    await stt.warm()  # silence: no words
    assert await stt.transcribe(b"\x00\x10" * 8, 16_000) == "w11 w12 w13"


async def test_kyutai_cancel_drops_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_moshi(monkeypatch)
    from agent_adapter.voice.kyutai import KyutaiSTT

    stt = KyutaiSTT()
    stream = stt.open(24_000)
    stream.feed(b"\x00\x10" * 8)
    await stream.cancel()
    assert (await stt.transcribe(b"\x00\x10" * 4, 24_000)).startswith("w")  # the model is free


@pytest.mark.parametrize(
    ("path", "name", "roots", "extra"),
    [
        ("whisper", "FasterWhisperSTT", ["faster_whisper"], "voice-whisper"),
        ("parakeet", "ParakeetSTT", ["nemo"], "voice-parakeet"),
        ("kyutai", "KyutaiSTT", ["moshi"], "voice-kyutai"),
        ("kyutai", "KyutaiTTS", ["moshi"], "voice-kyutai"),
        ("chatterbox_tts", "ChatterboxTTS", ["chatterbox"], "voice-chatterbox"),
        ("orpheus", "OrpheusTTS", ["orpheus_tts"], "voice-orpheus"),
        ("moshi_talker", "MoshiTalker", ["moshi"], "voice-kyutai"),
        ("haru", "HaruTalker", ["haru"], "pip install -e ./haru"),
    ],
)
def test_a_missing_library_names_its_install_command(
    monkeypatch: pytest.MonkeyPatch, path: str, name: str, roots: list[str], extra: str
) -> None:
    for root in roots:
        monkeypatch.setitem(sys.modules, root, None)
    cls = getattr(__import__(f"agent_adapter.voice.{path}", fromlist=[name]), name)
    kwargs = {"voice": "me.wav"} if name == "ChatterboxTTS" else {}
    with pytest.raises(MissingDependency, match=extra.replace(".", r"\.")):
        cls(**kwargs)
