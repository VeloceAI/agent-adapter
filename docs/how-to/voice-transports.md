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
| Raw audio and your own speech to text | `SilenceTurns` (or your VAD) feeding `EventPump` |
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

`SilenceTurns` reads 16-bit mono PCM and says when a turn starts and ends; Silero VAD or
Smart Turn fit the same `TurnDetector` port (`feed(pcm) -> ["started" | "ended"]`).

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
