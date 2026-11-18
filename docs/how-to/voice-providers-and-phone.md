---
title: "Hosted speech, echo and phone calls for the voice pipeline"
description: "Plug Deepgram, OpenAI, ElevenLabs or Cartesia into VoiceSession, play on laptop speakers without echo cutting the agent, and take keypad digits, hang up, transfer or place calls on Twilio."
---

# Hosted speech, echo and phone calls

This page adds three things to [our own voice pipeline](voice-pipeline.md): hosted
speech providers, a guard against the agent hearing itself on speakers, and the phone
basics on Twilio.

```bash
pip install "agent-adapter[voice-cloud]"  # httpx and websockets
```

## Use a hosted speech provider

Each class fits the `STT`, `StreamingSTT` or `TTS` port, so it drops into `VoiceSession`:

```python
from agent_adapter.voice import VoiceSession
from agent_adapter.voice.cloud import CartesiaTTS, ElevenLabsTTS, OpenAISTT, OpenAITTS
from agent_adapter.voice.deepgram import DeepgramSTT

session = VoiceSession(
    transport,
    stt=DeepgramSTT("nova-3"),  # streams each turn as it is spoken
    tts=ElevenLabsTTS("your-voice-id", sample_rate=24_000),
    turns=turns,
)
# or: stt=OpenAISTT(), tts=OpenAITTS(voice="coral"), tts=CartesiaTTS("your-voice-id")
```

| Class | Key from | What it calls |
|---|---|---|
| `DeepgramSTT` | `DEEPGRAM_API_KEY` | the live WebSocket per turn (`Finalize`, then `CloseStream` at its end); one `POST /v1/listen` for a whole turn |
| `OpenAISTT` | `OPENAI_API_KEY` | `POST /v1/audio/transcriptions` with the turn as WAV |
| `OpenAITTS` | `OPENAI_API_KEY` | `POST /v1/audio/speech`, raw 24 kHz PCM as it streams |
| `ElevenLabsTTS` | `ELEVENLABS_API_KEY` | `POST /v1/text-to-speech/{voice}/stream`, `output_format=pcm_<rate>` |
| `CartesiaTTS` | `CARTESIA_API_KEY` | `POST /tts/bytes`, raw `pcm_s16le` |

For the WebSocket speed features (Deepgram Flux early end of turn, text streamed into
Cartesia, ElevenLabs, Fish Audio or Rime, and AssemblyAI, Soniox, Cartesia Ink or Scribe
speech-to-text), see [Fast hosted voices](voxcat-providers.md).

- Pass `api_key=` to skip the environment. The key travels only in a request header; it
  never reaches a log or an error message.
- Failures raise the same typed errors as the model ports: `ModelAuthError`,
  `ModelRateLimited` (with `retry_after`), `ModelUnavailable` and `ModelTimeout`. A failed
  speech-to-text pass resumes playback; a broken Deepgram stream falls back to the whole turn.
- `OpenAISTT` and `OpenAITTS` take `base_url=` for any OpenAI-compatible server.
- `client=` takes your own `httpx.AsyncClient` (proxies, retries, tests); `aclose()` closes
  only a client the object made itself.

## Play on speakers without cutting yourself off

With no echo cancelling, the mic hears the agent and the agent stops for its own voice.
`EchoGuard` raises the bar only while the agent is speaking:

```python
from agent_adapter.voice import EchoGuard, LocalAudio, VoiceSession

guard = EchoGuard(level=0.1, loud_ms=200, speech_ms=1200)
session = VoiceSession(LocalAudio(), stt=stt, tts=tts, turns=turns, echo_guard=guard)
```

| While the agent speaks | What happens |
|---|---|
| Speech under `level` RMS that ends before `speech_ms` | nothing: playback goes on and nothing is transcribed |
| `loud_ms` of audio at `level` or above | a normal barge-in: playback pauses and the turn is heard |
| Speech that lasts `speech_ms` | a barge-in too, however quiet |

When the agent is quiet the guard does nothing. Set `level` between the agent's echo and
your voice at the mic; headphones or a browser (which cancels echo) need no guard. A real
echo canceller that installs from wheels on Windows and Linux exists in LiveKit's
`livekit` package (Apache-2.0, `AudioProcessingModule`); the pipeline does not use it yet.

## Phone basics on Twilio

```python
from agent_adapter.voice import TwilioCalls, TwilioStream, VoiceSession
from agent_adapter.voice.socket import text_socket

calls = TwilioCalls()  # TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN (or an API key)


@app.websocket("/twilio")
async def twilio(ws):
    await ws.accept()
    phone = TwilioStream(text_socket(ws.send_text, ws.iter_text()), calls=calls)
    session = VoiceSession(phone, stt=stt, tts=tts, turns=turns)
    await session.run(VoiceAgent(model, executor, session.speaker))
```

- **Keypad**: each `dtmf` digit goes to `phone.on_digit`. `VoiceSession` stops the agent at
  the first digit and hands it `(keypad) 1234` after `keypad_wait_ms` (1500) with no new
  digit, or at once on `#`. Change the words with `keypad_text=`.
- **Hang up**: `await phone.hang_up()` ends the call through the REST API (`Status=completed`)
  and ends `frames()`. With no `calls`, only our side ends and Twilio runs the TwiML after
  `<Connect>`, which hangs up when there is none.
- **Transfer**: `await phone.transfer("+15558675310", caller_id="+15551112222")` dials a
  person from the live call; Twilio ends the stream as it dials.
- **Call out**: `await calls.call(to, from_, twiml=stream_twiml("wss://you.example/twilio"))`
  places a call that connects to the same route when answered (`stream_twiml` is in
  `agent_adapter.voice.twilio_calls`).

Each of these has a side effect. When the agent may trigger one, wrap it in a tool marked
`confirm=True`, so a person (or the caller, with `VoiceApproval`) approves it first:

```python
@registry.tool(confirm=True)
async def transfer_to_desk(reason: str) -> str:
    """Hand the caller to a person at the front desk."""
    await current_call().transfer("+15558675310")  # look up this call's TwilioStream
    return "transferring"
```

Twilio's own errors raise the same typed errors (`provider="twilio"`); the credentials go
only in HTTP Basic auth. Test all of it offline: pass `client=` an
`httpx.AsyncClient(transport=httpx.MockTransport(handler))`, and `DeepgramSTT(connect=)` a
fake socket.

## Other phone networks

`TelnyxStream`, `PlivoStream` and `VonageStream` work like `TwilioStream`: paced playout,
a cut clears what is queued, each line is confirmed by the network, keypad presses reach
the agent and `hang_up()` ends our side.

| Network | Transport | Start it with | Audio |
|---|---|---|---|
| Telnyx | `TelnyxStream(text_socket(...))` | Call Control `stream_url` with `stream_bidirectional_mode="rtp"`, or TeXML `<Stream bidirectionalMode="rtp">` | 8 kHz μ-law; `mark` per line |
| Plivo | `PlivoStream(text_socket(...), content_type=...)` | `<Stream bidirectional="true">` | 8 or 16 kHz PCM, or μ-law; `checkpoint` per line |
| Vonage | `VonageStream(socket, rate=16000)` | NCCO `connect` to a WebSocket, `audio/l16;rate=16000` | 8, 16 or 24 kHz PCM; `notify` per line |

These are tested against each network's documented message formats with fake sockets,
not on live calls: run one test call on your account before going live.
