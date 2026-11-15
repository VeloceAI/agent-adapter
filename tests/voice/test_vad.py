"""Silero and Smart Turn adapters: the turn logic on scripted model output, and the real models
when Pipecat's bundled copies are installed."""

from pathlib import Path
from typing import ClassVar

import pytest

np = pytest.importorskip("numpy")
ort = pytest.importorskip("onnxruntime")

from agent_adapter.voice.silero import SileroTurns  # noqa: E402
from agent_adapter.voice.smart_turn import RATE, SmartTurn, log_mel  # noqa: E402
from agent_adapter.voice.turns import TurnEvent  # noqa: E402


class FakeSession:
    """Stands in for `onnxruntime.InferenceSession`, answering from a script of scores."""

    scores: ClassVar[list[float]] = []

    def __init__(self, path: str, **_: object) -> None:
        self.inputs: list[dict[str, object]] = []

    def run(self, _outputs: object, feeds: dict[str, object]) -> list[object]:
        self.inputs.append(feeds)
        score = np.array([[FakeSession.scores.pop(0)]], dtype=np.float32)
        if "state" in feeds:
            return [score, np.ones((2, 1, 128), dtype=np.float32)]
        return [score]


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    return FakeSession


def _windows(n: int, size: int = 512) -> bytes:
    return b"\x01\x00" * size * n


def _feed(detector: object, chunks: list[bytes]) -> list[TurnEvent]:
    return [event for chunk in chunks for event in detector.feed(chunk)]  # type: ignore[attr-defined]


def test_silero_turns_follow_speech_scores_with_hysteresis(fake):
    # 32 ms windows: 2 speech windows start a turn; 0.4 stays speech once speaking; 19 quiet end it
    fake.scores = [0.1, 0.9, 0.9, 0.4, 0.4] + [0.1] * 19
    turns = SileroTurns("model.onnx", start_ms=60, end_ms=600)
    events = _feed(turns, [_windows(1)[:-1], _windows(1)[-1:], _windows(23)])
    assert events == ["started", "ended"]
    first = turns._session.inputs[0]
    assert first["input"].shape == (1, 576) and int(first["sr"]) == 16_000


def test_silero_resets_its_memory_after_long_quiet(fake):
    fake.scores = [0.0] * 64
    turns = SileroTurns("model.onnx")
    _feed(turns, [_windows(64)])
    states = [float(feeds["state"].sum()) for feeds in turns._session.inputs]
    assert states[62] != 0.0 and states[63] == 0.0  # 63 windows of 32 ms pass 2 s of quiet


def test_silero_takes_8khz_and_rejects_other_rates(fake):
    fake.scores = [0.0]
    assert SileroTurns("m", sample_rate=8_000).feed(_windows(1, 256)) == []
    with pytest.raises(ValueError):
        SileroTurns("m", sample_rate=44_100)
    with pytest.raises(ValueError):
        SileroTurns("m", threshold=0.1)


class Script:
    """A detector that returns scripted events, one list per frame."""

    def __init__(self, *events: list[TurnEvent]) -> None:
        self.events = list(events)

    def feed(self, pcm: bytes) -> list[TurnEvent]:
        return self.events.pop(0) if self.events else []


FRAME = b"\x00\x00" * 320  # 20 ms


def test_smart_turn_ends_a_turn_only_when_the_model_says_it_is_done(fake):
    fake.scores = [0.2, 0.9]
    detector = Script(["started"], ["ended"], ["started"], ["ended"])
    turn = SmartTurn("model.onnx", detector)
    assert _feed(turn, [FRAME] * 4) == ["started", "ended"]
    features = turn._session.inputs[0]["input_features"]
    assert features.shape == (1, 80, 800) and features.dtype == np.float32


def test_smart_turn_ends_after_max_wait_of_quiet(fake):
    fake.scores = [0.1]
    turn = SmartTurn("model.onnx", Script(["started"], ["ended"]), max_wait_ms=100)
    assert _feed(turn, [FRAME] * 6) == ["started", "ended"]
    assert turn.last_probability == pytest.approx(0.1) and not turn.speaking


def test_smart_turn_rejects_bad_settings(fake):
    with pytest.raises(ValueError):
        SmartTurn("m", Script(), threshold=1.0)


def test_log_mel_shape_and_range():
    features = log_mel(np.random.default_rng(0).normal(0, 0.1, RATE * 8), np)
    assert features.shape == (80, 800)
    assert features.max() - features.min() <= 2.0 + 1e-6  # clamped to 8 decades, then / 4


def _pipecat_models() -> tuple[Path, Path]:
    pipecat = pytest.importorskip("pipecat")
    base = Path(pipecat.__file__).parent
    smart = sorted((base / "audio/turn/smart_turn/data").glob("smart-turn-v3*cpu*.onnx"))
    vad = base / "audio/vad/data/silero_vad.onnx"
    if not smart or not vad.exists():
        pytest.skip("Pipecat's bundled models are not installed")
    return vad, smart[-1]


@pytest.mark.contract
def test_real_models_stay_quiet_on_silence_and_match_pipecat_features():
    vad, smart = _pipecat_models()
    from pipecat.audio.turn.smart_turn._whisper_features import (
        compute_whisper_log_mel_features,
    )

    quiet = (np.random.default_rng(1).normal(0, 30, RATE * 2)).astype(np.int16).tobytes()
    assert SileroTurns(vad).feed(quiet) == []
    audio = np.random.default_rng(2).normal(0, 0.1, RATE * 8).astype(np.float32)
    assert np.array_equal(log_mel(audio, np), compute_whisper_log_mel_features(audio))
    assert 0.0 <= SmartTurn(smart, SileroTurns(vad)).probability(audio[: RATE * 2]) <= 1.0
