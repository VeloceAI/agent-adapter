from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _Stream:
    def __init__(self, update_interval: float) -> None:
        self.update_interval = update_interval
        self.audio: list[float] = []

    def start(self) -> None:
        pass

    def add_audio(self, audio: list[float], rate: int) -> None:
        self.audio += audio

    def stop(self) -> Any:
        line = types.SimpleNamespace(text=f"{len(self.audio)} samples")
        return types.SimpleNamespace(lines=[line])


class _Transcriber:
    def __init__(self, **kwargs: Any) -> None:
        self.streams: list[_Stream] = []

    def create_stream(self, update_interval: float) -> _Stream:
        self.streams.append(_Stream(update_interval))
        return self.streams[-1]


@pytest.fixture
def fake_moonshine(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("moonshine_voice")
    module.get_model_for_language = lambda language, arch: ("path", arch)  # type: ignore[attr-defined]
    module.string_to_model_arch = lambda name: name  # type: ignore[attr-defined]
    module.Transcriber = _Transcriber  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "moonshine_voice", module)


@pytest.mark.parametrize("interval", [None, 0.3])
async def test_stream_decodes_at_the_chosen_interval(
    fake_moonshine: None, interval: float | None
) -> None:
    from agent_adapter.voice.moonshine import MoonshineSTT

    stt = MoonshineSTT() if interval is None else MoonshineSTT(update_interval=interval)
    stream = stt.open(16_000)
    stream.feed(b"\x00\x01" * 160)
    assert await stream.finish() == "160 samples"
    assert stt._transcriber.streams[0].update_interval == (interval or 1.0)
