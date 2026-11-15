---
title: "Voice over a browser socket, or with OpenAI Realtime and Gemini Live"
description: "Run a safe voice agent over one WebSocket from a browser or phone gateway, detect turns from raw audio, or let OpenAI Realtime and Gemini Live call your tools without double bookings."
---

# Voice transports and realtime models

[Build a voice agent](voice-agents.md) covers the rules. This page connects them to audio:
a browser or phone gateway over one WebSocket, raw audio with turn detection, or a
speech-to-speech model that hears and speaks by itself.

| You have | Use |
|---|---|
| A browser, app or phone gateway that does speech to text and text to speech | `VoiceSocket` |
| Raw audio and your own speech to text | `SilenceTurns`, `SileroTurns` or `SmartTurn` feeding `EventPump` |
| OpenAI Realtime or Gemini Live | `RealtimeAgent` with `OpenAIRealtime()` or `GeminiLive(model)` |
| LiveKit or Pipecat | [LiveKit](../integrations/livekit.md), [Pipecat](../integrations/pipecat.md) |

## A browser or gateway over one WebSocket

The client hears and speaks; this side decides. Messages are small JSON objects:

| Client sends | Server sends |
|---|---|
| `{"type": "speech_started"}` | `{"type": "say", "id": 3, "text": "..."}` |
| `{"type": "transcript", "text": "...", "final": true}` | `{"type": "stop"}` |
| `{"type": "played", "id": 3}` once line 3 finished | |
| `{"type": "hangup"}` | |

```python
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.socket import VoiceSocket, text_socket


@app.websocket("/voice")  # FastAPI or Starlette
async def voice(ws):
    await ws.accept()
    channel = VoiceSocket(text_socket(ws.send_text, ws.iter_text()))
    await channel.run(VoiceAgent(model, executor, channel.speaker))
```

A line counts as heard only after its `played` message, so after a cut the history holds
exactly what the caller heard. A `websockets` connection is a `TextSocket` as it is.

## Raw audio: turn detection

Each detector reads 16-bit mono PCM and says when a turn starts and ends, behind one
`TurnDetector` port (`feed(pcm) -> ["started" | "ended"]`).

| Detector | Hears | Needs |
|---|---|---|
| `SilenceTurns()` | loudness | nothing |
| `SileroTurns(model_path)` | speech, not noise | `pip install "agent-adapter[vad]"`, `silero_vad.onnx` (MIT) |
| `SmartTurn(model_path, SileroTurns(...))` | whether the caller has finished, from how they spoke | the same, plus `smart-turn-v3` ONNX (BSD-2-Clause) |

Both models run locally on the CPU with ONNX Runtime; Pipecat ships copies of both files.
`SmartTurn` asks the model at each pause: "so maybe we could, um" keeps the turn open, and
it ends anyway after `max_wait_ms` (3 s) of quiet. Its features match Pipecat's exactly.

```python
from agent_adapter.voice import EventPump
from agent_adapter.voice.turns import SilenceTurns

turns, pump = SilenceTurns(end_ms=600), EventPump(agent)
for event in turns.feed(frame):
    if event == "started":
        pump.started_speaking()
    else:
        pump.said(await my_stt.final_text())
```

`EventPump` keeps events in order and never blocks your audio loop, even while a cut waits
for a booking to finish.

Measured with `python scripts/turn_check.py` (synthetic speech, then a laptop microphone
hearing the same speech from its speakers):

| Case | Loudness | Silero | Silero + Smart Turn |
|---|---|---|---|
| Two sentences, quiet room | both found | both found | both found |
| Two sentences, loud hiss | stuck "speaking" | both found | both found |
| "so maybe we could, um", then quiet | ended at the pause | ended at the pause | held, ended after 3 s |
| Live microphone, three sentences | all found | all found | all found, "um" held |

On real human speech, `python scripts/turn_bench.py --data DIR --fetch 300` scores them on
300 English human clips (65 MB) from Pipecat's Smart Turn v3.2 test set (CC-BY-4.0). Half
stop where the speaker finished, half stop mid-thought, then quiet follows. A false cut
ends a turn the speaker had not finished: they spoke again later in the clip, or a
mid-thought clip ended within the pause. Delay runs from the last speech to "ended".

| Detector | Cut, then spoke again | Cut in a 1 s pause | Cut in a 2 s pause | Delay p50 / p90 | Over 1.5 s late |
|---|---|---|---|---|---|
| `SilenceTurns()` (600 ms) | 19.3% | 80.0% | 98.0% | 504 / 1496 ms | 9.3% |
| `SileroTurns()` (600 ms) | 11.3% | 99.3% | 100% | 632 / 692 ms | 0% |
| `SmartTurn(SileroTurns())` | 8.7% | 14.7% | 15.3% | 632 / 752 ms | 2.7% |
| `SileroTurns(end_ms=1000)` | 1.3% | 5.3% | 100% | 1048 / 1132 ms | 4.0% |
| `SilenceTurns(end_ms=1000)` | 3.3% | 60.0% | 94.7% | 908 / 1888 ms | 24.3% |

No detector missed a turn, apart from loudness at 1 s (0.7%). Smart Turn holds a pause
mid-thought where a timer cannot, without slowing down most finished turns. Silero at 1 s
only looks good at a 1 s pause because it ends at about 1.05 s.

!!! note "Read these numbers with care"
    This is Smart Turn's own held-out test set, so it has a home advantage. Three in four
    clips come from one contributor (Liva AI). Speech end is where Silero last heard speech,
    the quiet after each clip is digital silence (kind to loudness), and how long the
    speaker really paused is unknown.

## OpenAI Realtime and Gemini Live

The model hears and speaks on its own; `RealtimeAgent` runs the tools it calls through your
executor. Open and authenticate the socket yourself, then pass it in.

```python
import websockets

from agent_adapter.voice.openai_realtime import OpenAIRealtime
from agent_adapter.voice.realtime import RealtimeAgent

url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
headers = {"Authorization": f"Bearer {key}"}
async with websockets.connect(url, additional_headers=headers) as socket:
    agent = RealtimeAgent(OpenAIRealtime(), socket, executor, instructions="You sell tickets.")
    await agent.run()
```

For Gemini, connect to the `BidiGenerateContent` endpoint and use `GeminiLive("gemini-3.8-live")`.

| When | What happens |
|---|---|
| Gemini cancels calls because the caller cut in | read-only tools stop; a tool that changes things finishes and the model is told "Already done" |
| The model calls the same tool and arguments again | the first result returns (`dedupe_window_s`, 120 s) |
| The caller cut in while a tool ran (OpenAI) | the result is sent, but no new reply is forced over the caller |
| A `confirm` tool | the prompt is spoken; only the caller's own transcribed words approve it |
| `minute_item="provider:name"` | the call's minutes are billed in the ledger when `run` ends |

!!! note "What is not checked live yet"
    The codecs follow the providers' published message formats and are tested with
    recorded-style messages, not live calls. Gemini has no "say exactly" message, so an
    approval prompt is a text turn asking the model to read it. Whether either provider
    accepts a tool result after a cut is not documented; this side never needs it to.
