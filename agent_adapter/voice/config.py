"""`VoiceConfig`: a whole VoxCat call from one YAML file or dict.

```yaml
stt: deepgram                      # a name below, or "package.module:Class"
tts: {use: cartesia, voice: "a0e99841-438c-4a64-b679-ae501e7d6091"}
model: {use: openai, model: gpt-4.1-mini}
instructions: You book tables. Keep answers short.
tools: mypkg.tools:registry        # optional
turns: {end_ms: 600, eager: true}  # eager: EagerTurns on SilenceTurns
early_ms: 100
speed: {first_words: 4, acks: ["One moment."]}
```

Keys stay in the environment: a secret-looking option (`api_key`, `token`) is refused.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_adapter.core.errors import ConfigError, MissingDependency
from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.voice.ack import Acknowledgements
from agent_adapter.voice.agent import VoiceAgent
from agent_adapter.voice.endpoint import EagerTurns
from agent_adapter.voice.media import AudioTransport
from agent_adapter.voice.session import VoiceSession
from agent_adapter.voice.speed import Speed
from agent_adapter.voice.turns import SilenceTurns, TurnDetector

_V = "agent_adapter.voice."
STT_NAMES = {
    "deepgram": _V + "deepgram:DeepgramSTT",
    "deepgram-flux": _V + "deepgram:DeepgramFluxSTT",
    "openai": _V + "cloud:OpenAISTT",
    "cartesia": _V + "cartesia:CartesiaInkSTT",
    "elevenlabs": _V + "elevenlabs:ElevenLabsScribeSTT",
    "assemblyai": _V + "assemblyai:AssemblyAISTT",
    "soniox": _V + "soniox:SonioxSTT",
    "moonshine": _V + "moonshine:MoonshineSTT",
    "whisper": _V + "whisper:FasterWhisperSTT",
    "parakeet": _V + "parakeet:ParakeetSTT",
    "kyutai": _V + "kyutai:KyutaiSTT",
}
TTS_NAMES = {
    "cartesia": _V + "cartesia:CartesiaLiveTTS",
    "elevenlabs": _V + "elevenlabs:ElevenLabsLiveTTS",
    "openai": _V + "cloud:OpenAITTS",
    "fish": _V + "fish:FishAudioTTS",
    "rime": _V + "rime:RimeTTS",
    "moonshine": _V + "moonshine:MoonshineTTS",
    "kyutai": _V + "kyutai:KyutaiTTS",
    "chatterbox": _V + "chatterbox_tts:ChatterboxTTS",
    "orpheus": _V + "orpheus:OrpheusTTS",
}
MODEL_NAMES = {
    "openai": "agent_adapter.models:OpenAIModel",
    "anthropic": "agent_adapter.models:AnthropicModel",
    "gemini": "agent_adapter.models:GeminiModel",
}
_SECRET = ("key", "token", "secret", "password")
_MAX_BYTES = 64_000


@dataclass(frozen=True)
class Part:
    """One piece of the call: `use` (a short name or "module:Name") and its options."""

    use: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def build(self, names: Mapping[str, str]) -> Any:
        target = load(names.get(self.use, self.use))
        return target(**self.options) if callable(target) else target


@dataclass(frozen=True)
class VoiceConfig:
    stt: Part
    tts: Part
    model: Part
    instructions: str = ""
    tools: str | None = None
    turns: Mapping[str, Any] = field(default_factory=dict)
    early_ms: float = 100
    speed: Mapping[str, Any] | None = field(default_factory=dict)  # None: no speed tricks

    @classmethod
    def load(cls, source: str | Path | Mapping[str, Any]) -> VoiceConfig:
        """From a dict, or a .yaml/.yml/.json file. Raises `ConfigError` on a bad setting."""
        data = source if isinstance(source, Mapping) else _read(Path(source))
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ConfigError(f"voice config: unknown keys {sorted(unknown)}")
        missing = [k for k in ("stt", "tts", "model") if k not in data]
        if missing:
            raise ConfigError(f"voice config: {', '.join(missing)} missing")
        speed = data.get("speed", {})
        return cls(
            stt=_part(data["stt"], "stt"),
            tts=_part(data["tts"], "tts"),
            model=_part(data["model"], "model"),
            instructions=str(data.get("instructions", "")),
            tools=data.get("tools"),
            turns=dict(data.get("turns") or {}),
            early_ms=float(data.get("early_ms", 100)),
            speed=None if speed is False else dict(speed or {}),
        )

    def turn_detector(self) -> TurnDetector:
        options = dict(self.turns)
        eager = options.pop("eager", True)
        done_ms = options.pop("done_ms", 250)
        turns = SilenceTurns(**options)
        return EagerTurns(turns, done_ms=done_ms) if eager else turns

    def build(
        self, transport: AudioTransport, executor: Executor | None = None
    ) -> tuple[VoiceSession, VoiceAgent]:
        """The session and agent for one call on `transport`; run `session.run(agent)`.

        Pass an `executor` with an approval port for `confirm=True` tools; the default one
        has none, so those calls are refused.
        """
        registry = load(self.tools) if self.tools else Registry()
        if not isinstance(registry, Registry):
            raise ConfigError(f"voice config: tools {self.tools!r} is not a Registry")
        session = VoiceSession(
            transport,
            stt=self.stt.build(STT_NAMES),
            tts=self.tts.build(TTS_NAMES),
            turns=self.turn_detector(),
            early_ms=self.early_ms,
        )
        agent = VoiceAgent(
            self.model.build(MODEL_NAMES),
            executor if executor is not None else Executor(registry),
            session.speaker,
            instructions=self.instructions,
            registry=registry,
            speed=self._speed(),
        )
        return session, agent

    def describe(self) -> list[str]:
        turns = "EagerTurns" if self.turns.get("eager", True) else "SilenceTurns"
        speed = (
            "off" if self.speed is None else ", ".join(f"{k}={v}" for k, v in self.speed.items())
        )
        return [
            f"stt: {STT_NAMES.get(self.stt.use, self.stt.use)}",
            f"tts: {TTS_NAMES.get(self.tts.use, self.tts.use)}",
            f"model: {MODEL_NAMES.get(self.model.use, self.model.use)}",
            f"tools: {self.tools or 'none'}",
            f"turns: {turns} {dict(self.turns) or ''}".rstrip(),
            f"early_ms: {self.early_ms:g}",
            f"speed: {speed or 'on'}",
        ]

    def _speed(self) -> Speed | None:
        if self.speed is None:
            return None
        options = dict(self.speed)
        acks = options.pop("acks", None)
        return Speed(acks=Acknowledgements(acks) if acks else None, **options)


def load(ref: str) -> Any:
    """`package.module:Name` -> the object."""
    module, _, name = ref.partition(":")
    if not module or not name:
        raise ConfigError(f"voice config: {ref!r} is not a known name or 'module:Name'")
    try:
        target: Any = importlib.import_module(module)
    except ImportError as exc:
        raise ConfigError(f"voice config: cannot import {module!r}: {exc}") from exc
    for part in name.split("."):
        try:
            target = getattr(target, part)
        except AttributeError:
            raise ConfigError(f"voice config: {module!r} has no {name!r}") from None
    return target


def _part(value: Any, key: str) -> Part:
    if isinstance(value, str):
        return Part(value)
    if not isinstance(value, Mapping) or not isinstance(value.get("use"), str):
        raise ConfigError(f"voice config: {key} needs a name or a mapping with 'use'")
    options = {k: v for k, v in value.items() if k != "use"}
    secret = [k for k in options if any(part in k.lower() for part in _SECRET)]
    if secret:
        raise ConfigError(
            f"voice config: {key}.{secret[0]} looks secret; set it in the environment"
        )
    return Part(value["use"], options)


def _read(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if len(text) > _MAX_BYTES:
        raise ConfigError(f"voice config: {path} is over {_MAX_BYTES} bytes")
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise MissingDependency("a YAML voice config needs pyyaml: pip install pyyaml") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ConfigError(f"voice config: {path} must hold a mapping")
    return data


__all__ = ["MODEL_NAMES", "STT_NAMES", "TTS_NAMES", "Part", "VoiceConfig", "load"]
