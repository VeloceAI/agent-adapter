import json
import logging
import runpy
from pathlib import Path

import pytest

from agent_adapter.errors import ModelError
from agent_adapter.voice.assemblyai import AssemblyAISTT
from agent_adapter.voice.cartesia import CartesiaLiveTTS
from agent_adapter.voice.soniox import SonioxSTT
from tests.voice.fake_ws import frames, pcm, serve

pytest.importorskip("msgpack")
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "voxcat_providers_offline.py"
KEY = "sk-live-must-never-leak-0123456789"


def test_the_offline_example_shows_turn_signals_and_adds_under_a_millisecond_per_chunk(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        "final: Book two seats.",
        "signal: eager_end 'book two' 0.55",
        "signal: resumed 'book two' 0.3",
        "signal: end 'Book two seats.' 0.9",
    ]
    rows = {line[:26].strip(): float(line.split()[-1]) for line in lines[5:]}
    assert set(rows) == {
        "cartesia sonic",
        "elevenlabs stream-input",
        "fish audio live",
        "rime v1",
        "deepgram flux",
        "assemblyai",
        "soniox",
        "cartesia ink",
        "elevenlabs scribe",
    }
    assert all(0 < us < 1000 for us in rows.values()), rows


async def test_a_key_never_reaches_a_log_an_error_or_a_url(caplog):
    caplog.set_level(logging.DEBUG)

    async def refuse(connection, seen):
        async for _ in frames(connection, seen):
            await connection.send(json.dumps({"error_code": 401, "error_message": "unauthorised"}))
            await connection.send(
                json.dumps({"type": "Error", "error_code": 1008, "error": "invalid"})
            )
            error = {"type": "error", "status_code": 401, "message": "denied"}
            await connection.send(json.dumps({**error, "context_id": json.loads(_)["context_id"]}))

    errors: list[str] = []
    async with serve(refuse) as (base, seen):
        for stt in (SonioxSTT(api_key=KEY, url=base), AssemblyAISTT(api_key=KEY, url=base)):
            stream = stt.open(16_000)
            stream.feed(pcm(1600))
            with pytest.raises(ModelError) as caught:
                await stream.finish()
            errors.append(repr(caught.value))
        with pytest.raises(ModelError) as caught:
            b"".join(
                [c async for c in CartesiaLiveTTS("v", api_key=KEY, url=base).synthesize("Hi")]
            )
        errors.append(repr(caught.value))
    assert len(errors) == 3
    ours = [r.getMessage() for r in caplog.records if r.name != "websockets.server"]  # the fake's
    assert any("Authorization: [redacted]" in line for line in ours)
    assert not any(KEY in line for line in ours) and not any(KEY in e for e in errors)
    assert KEY not in seen.path
