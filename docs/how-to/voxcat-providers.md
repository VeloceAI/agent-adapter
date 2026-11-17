---
title: "Fast hosted voices for VoxCat: Deepgram Flux, Cartesia, ElevenLabs, Fish Audio, AssemblyAI, Soniox, Rime"
description: "Plug the fastest hosted speech providers into our voice pipeline over WebSocket: early end of turn, text streamed into speech while the model writes, warm sockets and word timings."
---

# Fast hosted voices for VoxCat

VoxCat is our own voice pipeline (`agent_adapter.voice`). This page adds the hosted
providers' speed features over WebSocket: the speech-to-text side says early that the
caller is probably done, and the text-to-speech side takes text while the model is still
writing it. For the plain HTTP providers and phone calls, see
[Hosted speech, echo and phone calls](voice-providers-and-phone.md).

```bash
pip install "agent-adapter[voice-cloud]"  # httpx, websockets, msgpack
```

## The providers

| Class | Kind | Speed features | Key from |
|---|---|---|---|
| `DeepgramFluxSTT` (`voice.deepgram`) | STT | `EagerEndOfTurn`, `TurnResumed`, `EndOfTurn`; `ForceEndTurn` at our end of turn | `DEEPGRAM_API_KEY` |
| `AssemblyAISTT` (`voice.assemblyai`) | STT | `end_of_turn`; `ForceEndpoint` at our end of turn | `ASSEMBLYAI_API_KEY` |
| `SonioxSTT` (`voice.soniox`) | STT | `<end>` endpoint token; `finalize` stops at `<fin>` | `SONIOX_API_KEY` |
| `CartesiaInkSTT` (`voice.cartesia`) | STT | `finalize` stops at `flush_done` | `CARTESIA_API_KEY` |
| `ElevenLabsScribeSTT` (`voice.elevenlabs`) | STT | VAD or turn-prediction commits; a manual commit at our end | `ELEVENLABS_API_KEY` |
| `CartesiaLiveTTS` (`voice.cartesia`) | TTS | one shared socket, a `context_id` per reply (`continue: true`, then `false`), word timestamps, speed, volume, emotion | `CARTESIA_API_KEY` |
| `ElevenLabsLiveTTS` (`voice.elevenlabs`) | TTS | `stream-input`: `chunk_length_schedule`, `flush`, word alignment, a warm next socket | `ELEVENLABS_API_KEY` |
| `FishAudioTTS` (`voice.fish`) | TTS | `/v1/tts/live` MessagePack frames, `flush`, a saved voice or a reference clip, a warm next socket | `FISH_AUDIO_API_KEY` |
| `RimeTTS` (`voice.rime`) | TTS | WebSocket v1 (Coda or Mist v3): one shared socket, a `contextId` per reply, word spans | `RIME_API_KEY` |

**How tested:** every adapter is format-tested against a local fake server that speaks
the provider's documented messages (read 2026-11-18); none was called live, since we hold
no keys. Run one real call on your account before going live.
**Licence:** each provider is a paid, proprietary hosted API under its own terms. Our
client code is Apache-2.0 and uses `websockets` (BSD-3-Clause), `msgpack` (Apache-2.0) and
`httpx` (BSD-3-Clause).

## Start a reply before the turn ends

A stream that has `on_turn` (the `TurnSignals` port in `agent_adapter.voice.ports`)
reports `TurnSignal(event, text, confidence)`: `eager_end` (probably done: start the reply
but keep it unspoken), `resumed` (the caller went on: drop it) and `end` (done).

```python
from agent_adapter.voice.deepgram import DeepgramFluxSTT
from agent_adapter.voice.ports import TurnSignal, TurnSignals

stt = DeepgramFluxSTT(eot_threshold=0.7, eager_eot_threshold=0.5)
stream = stt.open(16_000)


def heard(signal: TurnSignal) -> None:
    print(signal.event, signal.text)  # start, drop or speak the reply here


if isinstance(stream, TurnSignals):
    stream.on_turn(heard)
for frame in frames:  # 16-bit mono PCM as it arrives
    stream.feed(frame)
text = await stream.finish()  # at once when EndOfTurn is already in
```

Every streaming STT here works without the callback too: the session then decides the
end of the turn on its own, as with `DeepgramSTT`.

## Stream text into speech while the model writes

A `StreamingTTS` has `open()`, which returns a `TTSStream`: `push` text as it comes
(`flush=True` at the end of a clause), `end()`, and read `audio()` as it arrives. It
still fits the plain `TTS` port, so `synthesize(text)` works everywhere.

```python
import asyncio

from agent_adapter.voice.cartesia import CartesiaLiveTTS

tts = CartesiaLiveTTS("your-voice-id", speed=1.1, emotion="calm", max_buffer_delay_ms=200)
await tts.warm()  # open the socket when the call starts
stream = tts.open()


async def write() -> None:
    async for piece in model_tokens():  # your model's text as it streams
        await stream.push(piece, flush=piece.endswith((".", "?", "!")))
    await stream.end()


writer = asyncio.create_task(write())
async for chunk in stream.audio():  # PCM16 at tts.sample_rate
    play(chunk)
await writer
print(stream.words)  # WordTime(word, start, end), when the provider sends timings
```

`stream.cancel()` stops a reply at once (a barge-in). On a shared socket (Cartesia,
Rime) the next reply reuses the socket; ElevenLabs and Fish open one socket per reply,
and `warm()` opens the next one ahead (a spare older than `spare_ttl_s`, 15 s, is
replaced).

```python
from agent_adapter.voice.elevenlabs import ElevenLabsLiveTTS
from agent_adapter.voice.fish import FishAudioTTS
from agent_adapter.voice.rime import RimeTTS

eleven = ElevenLabsLiveTTS("voice-id", chunk_length_schedule=(50, 120, 160, 290))
fish = FishAudioTTS(reference=(clip_wav, "words said in the clip"), latency="balanced")
rime = RimeTTS("lyra", model="coda", time_scale=0.95)
```

## Safety and errors

- Each key is read from its environment variable on first use (or `api_key=`) and sent
  only in a header, never in a URL or a message body; Soniox gets it as a Bearer header
  instead of in its config message. The `websockets` debug log of the handshake shows
  `Authorization`, `X-API-Key` and `xi-api-key` as `[redacted]`.
- Failures raise the typed model errors: `ModelAuthError`, `ModelRateLimited`,
  `ModelUnavailable` (a dropped socket) and `ModelError` with the provider's message.
- `connect=` takes your own `connect(url, headers)` coroutine, for tests or a proxy;
  `url=` points an adapter at another region or a fake server.

## Measure our own overhead

`examples/voxcat_providers_offline.py` runs every adapter against an in-memory fake, with
no keys and no network. It prints Flux's turn signals and the time our code adds per
audio chunk (the provider's own delay is not in it):

```bash
python examples/voxcat_providers_offline.py
```

On a laptop each adapter adds about 3 to 20 microseconds per chunk; ElevenLabs Scribe
adds about 50, because its audio goes out as base64 inside JSON.

## Not built yet

ElevenLabs' `multi-stream-input` (one socket for many replies), Deepgram Flux `Configure`
mid-call, AssemblyAI `UpdateConfiguration`, and Rime's legacy `/ws2` and `/ws3` sockets.
