import io
import json

import pytest

from agent_adapter import CallableModel, Registry
from agent_adapter.cli.main import main
from agent_adapter.errors import ConfigError
from agent_adapter.voice.config import Part, VoiceConfig, load
from agent_adapter.voice.endpoint import EagerTurns
from agent_adapter.voice.testing import MemoryTransport, ScriptedSTT, WordTTS
from agent_adapter.voice.turns import SilenceTurns

registry = Registry()


@registry.tool
def opening_hours() -> str:
    """When the restaurant is open."""
    return "17 to 23"


def stt() -> ScriptedSTT:
    return ScriptedSTT(["hello"])


def tts() -> WordTTS:
    return WordTTS()


model = CallableModel(lambda request: "Hi.", model="config-test")
HERE = "tests.voice.test_voice_config"
CONFIG = {
    "stt": f"{HERE}:stt",
    "tts": {"use": f"{HERE}:tts"},
    "model": f"{HERE}:model",
    "instructions": "Be brief.",
    "tools": f"{HERE}:registry",
    "turns": {"end_ms": 700},
    "speed": {"first_words": 3, "acks": ["One moment."]},
}


def test_a_config_builds_a_whole_call():
    session, agent = VoiceConfig.load(CONFIG).build(MemoryTransport())
    assert isinstance(session.stt, ScriptedSTT) and isinstance(session.speaker.tts, WordTTS)
    assert isinstance(session.turns, EagerTurns) and session.turns.base_ms == 700
    assert session.early_ms == 100 and agent.model is model
    assert "opening_hours" in agent.registry.names()
    assert agent.speed is not None and agent.speed.first_words == 3
    assert agent.speed.acks is not None


def test_plain_turns_and_no_speed_tricks():
    config = VoiceConfig.load({**CONFIG, "turns": {"eager": False}, "speed": False})
    session, agent = config.build(MemoryTransport())
    assert isinstance(session.turns, SilenceTurns) and agent.speed is None
    assert "speed: off" in config.describe()


def test_short_names_point_at_our_providers():
    config = VoiceConfig.load({"stt": "deepgram", "tts": "cartesia", "model": "openai"})
    lines = config.describe()
    assert "stt: agent_adapter.voice.deepgram:DeepgramSTT" in lines
    assert "tts: agent_adapter.voice.cartesia:CartesiaLiveTTS" in lines
    assert "model: agent_adapter.models:OpenAIModel" in lines


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"stt": "x:y"}, "tts, model missing"),
        ({**CONFIG, "voices": 2}, "unknown keys"),
        ({**CONFIG, "stt": {"use": "deepgram", "api_key": "dg-1"}}, "looks secret"),
        ({**CONFIG, "tts": {"model": "sonic"}}, "needs a name"),
        ({**CONFIG, "tools": f"{HERE}:model"}, "is not a Registry"),
    ],
)
def test_bad_configs_are_refused(data, message):
    with pytest.raises(ConfigError, match=message):
        VoiceConfig.load(data).build(MemoryTransport())


@pytest.mark.parametrize(
    ("ref", "message"),
    [
        ("nothing", "not a known name"),
        ("no_such_mod_xyz:A", "cannot import"),
        (f"{HERE}:nope", "has no"),
    ],
)
def test_load_names_what_is_wrong(ref, message):
    with pytest.raises(ConfigError, match=message):
        load(ref)


def test_a_part_that_is_not_callable_is_used_as_is():
    assert Part(f"{HERE}:model").build({}) is model


def test_the_cli_checks_a_json_config(tmp_path):
    path = tmp_path / "voxcat.json"
    path.write_text(json.dumps(CONFIG), encoding="utf-8")
    out = io.StringIO()
    assert main(["voice", "check", str(path)], out) == 0
    assert f"stt: {HERE}:stt" in out.getvalue() and "turns: EagerTurns" in out.getvalue()


def test_the_cli_reads_yaml_and_refuses_a_bad_file(tmp_path, capsys):
    pytest.importorskip("yaml")
    good = tmp_path / "voxcat.yaml"
    good.write_text(f"stt: {HERE}:stt\ntts: {HERE}:tts\nmodel: {HERE}:model\n", encoding="utf-8")
    assert main(["voice", "check", str(good)], io.StringIO()) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a list\n", encoding="utf-8")
    assert main(["voice", "check", str(bad)], io.StringIO()) == 2
    assert "must hold a mapping" in capsys.readouterr().err
