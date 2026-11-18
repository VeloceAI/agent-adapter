"""Stand-ins for the open speech libraries, shaped after the calls in their READMEs and
model cards. No model, GPU, NumPy or PyTorch is needed."""

from __future__ import annotations

import contextlib
import sys
import threading
import types
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest


class T:
    """Stands in for a NumPy array or a torch tensor: flat values, and a token for indexing."""

    def __init__(self, values: Any = (), *, token: Any = None, dtype: Any = None) -> None:
        self.values, self.token, self.dtype = [float(v) for v in values], token, dtype

    def reshape(self, *shape: int) -> T:
        return self

    def tolist(self) -> list[float]:
        return self.values

    def detach(self) -> T:
        return self

    def float(self) -> T:
        return self

    def cpu(self) -> T:
        return self

    def item(self) -> Any:
        return self.token

    def all(self) -> bool:
        return bool(self.token)

    def __getitem__(self, key: Any) -> T:
        return self

    def __ne__(self, other: object) -> T:  # type: ignore[override]
        return T(token=self.token)  # a frame's token says whether it is complete

    def __len__(self) -> int:
        return len(self.values)


def module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, mod)
    return mod


def fake_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    module(monkeypatch, "numpy", asarray=lambda values, dtype=None: T(values, dtype=dtype))


def fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    module(
        monkeypatch,
        "torch",
        tensor=lambda values, dtype=None, device=None: T(values, dtype=dtype),
        float32="float32",
        bfloat16="bfloat16",
        no_grad=contextlib.nullcontext,
    )


class Loads:
    """Counts model loads, to check that a model is loaded once and shared."""

    count = 0


# faster-whisper -----------------------------------------------------------------------


class WhisperModel:
    def __init__(self, size: str, device: str = "auto", compute_type: str = "default") -> None:
        Loads.count += 1
        self.args = (size, device, compute_type)
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: T, **options: Any) -> tuple[Iterator[Any], Any]:
        self.calls.append(options)

        def segments() -> Iterator[Any]:
            yield types.SimpleNamespace(text=f" {len(audio)} samples ")
            yield types.SimpleNamespace(text=f" {audio.dtype}")

        return segments(), types.SimpleNamespace(language="en")


# NeMo Parakeet ------------------------------------------------------------------------


class ASRModel:
    def __init__(self, name: str) -> None:
        Loads.count += 1
        self.name, self.device, self.evaluated, self.calls = name, None, False, []
        self.legacy = False

    @classmethod
    def from_pretrained(cls, model_name: str) -> ASRModel:
        return cls(model_name)

    def to(self, device: str) -> ASRModel:
        self.device = device
        return self

    def eval(self) -> None:
        self.evaluated = True

    def transcribe(self, audio: list[T], batch_size: int = 4, verbose: bool = True) -> list[Any]:
        self.calls.append((batch_size, verbose))
        text = f"{len(audio[0])} samples"
        return [text if self.legacy else types.SimpleNamespace(text=text)]


def fake_nemo(monkeypatch: pytest.MonkeyPatch) -> None:
    module(monkeypatch, "nemo")
    module(monkeypatch, "nemo.collections")
    module(monkeypatch, "nemo.collections.asr", models=types.SimpleNamespace(ASRModel=ASRModel))


# Chatterbox ---------------------------------------------------------------------------


class Chatterbox:
    sr = 24_000

    def __init__(self, **load: Any) -> None:
        Loads.count += 1
        self.load, self.calls = load, []

    @classmethod
    def from_pretrained(cls, **load: Any) -> Chatterbox:
        return cls(**load)

    def generate(self, text: str, **options: Any) -> T:
        self.calls.append((text, options))
        return T([0.5] * len(text))


def fake_chatterbox(monkeypatch: pytest.MonkeyPatch) -> None:
    module(monkeypatch, "chatterbox")
    module(monkeypatch, "chatterbox.tts", ChatterboxTTS=type("ChatterboxTTS", (Chatterbox,), {}))
    turbo = type("ChatterboxTurboTTS", (Chatterbox,), {})
    module(monkeypatch, "chatterbox.tts_turbo", ChatterboxTurboTTS=turbo)
    multi = type("ChatterboxMultilingualTTS", (Chatterbox,), {})
    module(monkeypatch, "chatterbox.mtl_tts", ChatterboxMultilingualTTS=multi)


# Orpheus ------------------------------------------------------------------------------


class OrpheusModel:
    last: OrpheusModel
    gate: threading.Event | None = None  # set: chunks after the first wait for it

    def __init__(self, model_name: str, max_model_len: int = 2048) -> None:
        Loads.count += 1
        OrpheusModel.last = self
        self.args, self.calls, self.closed = (model_name, max_model_len), [], False

    def generate_speech(self, **request: Any) -> Iterator[bytes]:
        self.calls.append(request)

        def chunks() -> Iterator[bytes]:
            try:
                for i in range(3):
                    if i and OrpheusModel.gate is not None:
                        OrpheusModel.gate.wait(5)
                    yield bytes([i + 1, 0]) * 4
            finally:
                self.closed = True

        return chunks()


# Kyutai and Moshi (the moshi package) ------------------------------------------------


class Mimi:
    frame_size, frame_rate, sample_rate = 4, 12.5, 24_000

    def __init__(self) -> None:
        self.streams, self.codebooks = 0, 32

    @contextlib.contextmanager
    def streaming(self, batch_size: int) -> Iterator[None]:
        self.streams += 1
        yield

    def set_num_codebooks(self, n: int) -> None:
        self.codebooks = n

    def encode(self, audio: T) -> T:
        return audio

    def decode(self, codes: T) -> T:
        return T([0.25 if codes.token not in (None, 3) else 0.0] * self.frame_size)


class LMGen:
    def __init__(self, lm: Any, temp: float = 0.8, temp_text: float = 0.7) -> None:
        self.lm, self.temps, self.steps, self.words = lm, (temp, temp_text), 0, 0

    @contextlib.contextmanager
    def streaming(self, batch_size: int) -> Iterator[None]:
        yield

    def step(self, codes: T) -> T | None:
        self.steps += 1
        if self.steps == 1:
            return None  # the model's first frame is still in its delay
        if not any(codes.values):
            return T(token=3)  # padding
        self.words += 1
        return T(token=10 + self.words)


class Tokenizer:
    def id_to_piece(self, token: int) -> str:
        return f"▁w{token}"


class CheckpointInfo:
    stt_config: ClassVar[dict[str, float]] = {
        "audio_silence_prefix_seconds": 0.08,
        "audio_delay_seconds": 0.16,
    }
    raw_config: ClassVar[dict[str, int]] = {"text_padding_token_id": 3}
    last: CheckpointInfo

    def __init__(self, repo: str) -> None:
        Loads.count += 1
        CheckpointInfo.last = self
        self.repo, self.mimi, self.lm_args = repo, Mimi(), {}

    @classmethod
    def from_hf_repo(cls, repo: str) -> CheckpointInfo:
        return cls(repo)

    def get_mimi(self, device: str) -> Mimi:
        return self.mimi

    def get_text_tokenizer(self) -> Tokenizer:
        return Tokenizer()

    def get_moshi(self, **args: Any) -> str:
        self.lm_args = args
        return "lm"


class TTSModel:
    last: TTSModel

    def __init__(self, info: CheckpointInfo, **args: Any) -> None:
        TTSModel.last = self
        self.info, self.args, self.mimi = info, args, Mimi()
        self.voices: list[str] = []
        self.gate: threading.Event | None = None

    @classmethod
    def from_checkpoint_info(cls, info: CheckpointInfo, **args: Any) -> TTSModel:
        return cls(info, **args)

    def prepare_script(self, texts: list[str], padding_between: int = 1) -> list[str]:
        return texts[0].split()

    def get_voice_path(self, name: str) -> str:
        return f"/voices/{name}.safetensors"

    def make_condition_attributes(self, paths: list[str], cfg_coef: float = 2.0) -> Any:
        self.voices.append(paths[0])
        return {"voice": paths[0], "cfg": cfg_coef}

    def generate(self, entries: list[list[str]], conditions: list[Any], on_frame: Any) -> None:
        on_frame(T(token=False))  # a frame still in the delay holds -1 codes: skipped
        for i, _word in enumerate(entries[0]):
            if i and self.gate is not None:
                self.gate.wait(5)
            on_frame(T(token=10 + i))


def fake_moshi(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch(monkeypatch)
    module(monkeypatch, "moshi")
    loaders = types.SimpleNamespace(CheckpointInfo=CheckpointInfo)
    module(monkeypatch, "moshi.models", loaders=loaders, LMGen=LMGen)
    module(monkeypatch, "moshi.models.loaders", CheckpointInfo=CheckpointInfo)
    module(monkeypatch, "moshi.models.tts", TTSModel=TTSModel)


# Haru-4B ------------------------------------------------------------------------------


class HaruModel:
    @classmethod
    def from_pretrained(cls, repo: str) -> HaruModel:
        Loads.count += 1
        return cls()


class DuplexSession:
    last: DuplexSession
    length = 4

    def __init__(self, model: HaruModel, **options: Any) -> None:
        DuplexSession.last = self
        self.options, self.heard = options, []

    def frame(self, loud: bool, text: str) -> Any:
        audio = T([0.5 if loud else 0.0] * 3_200)
        return types.SimpleNamespace(audio=audio, text=text, done=len(self.heard) >= self.length)

    def start(self) -> Any:
        return self.frame(False, "")

    def step(self, user_audio: T) -> Any:
        self.heard.append(user_audio)
        loud = any(user_audio.values)
        return self.frame(loud, "mm-hm" if loud else "")


def fake_haru(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    fake_numpy(monkeypatch)
    prepared: list[str] = []

    def prepare_reference(audio: str) -> str:
        prepared.append(audio)
        return f"ref:{audio}"

    module(monkeypatch, "haru", HaruModel=HaruModel, DuplexSession=DuplexSession)
    module(
        monkeypatch,
        "haru.audio",
        load_audio=lambda p: f"wav:{p}",
        prepare_reference=prepare_reference,
    )
    return prepared
