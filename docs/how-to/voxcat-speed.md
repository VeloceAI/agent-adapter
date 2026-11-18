---
title: "Faster voice replies: speculative replies, early tools, acknowledgements and cached answers"
description: "Make VoxCat (VoiceSession and VoiceAgent) answer sooner: start the reply on the early transcript, run opted-in read-only tools early, play a pre-rendered acknowledgement, replay cached answers with audio and warm up at call start."
---

# Answer sooner on a voice call

`Speed` makes a [`VoiceAgent`](voice-agents.md) answer sooner. Each part is optional:

| Part | What it does |
|---|---|
| speculative reply | starts the reply on the early transcript; keeps it if the final one has the same words |
| `SpeculativeTools` | lets that reply run opted-in read-only tools before the turn is final |
| `Acknowledgements` | plays a short pre-rendered line ("Mm-hm.") when the reply will be slow |
| `AnswerCache` | replays the reply to a repeated question, audio included |
| `Warmup` | connects, renders lines and prefetches read-only tools at call start |

```python
from agent_adapter.voice import VoiceAgent, VoiceSession
from agent_adapter.voice.ack import Acknowledgements
from agent_adapter.voice.answers import AnswerCache
from agent_adapter.voice.speculate import SpeculativeTools
from agent_adapter.voice.speed import Speed
from agent_adapter.voice.warm import Warmup

answers = AnswerCache(ttl_s=300)  # one per process, shared by its calls
session = VoiceSession(transport, stt=stt, tts=answers.voice(tts), turns=turns, early_ms=200)
speed = Speed(
    tools=SpeculativeTools(["weather"]),
    acks=Acknowledgements(["Mm-hm.", "Okay."]),
    answers=answers,
    tenant="acme",
    warmup=Warmup(connections=[pool], tools=[("account", {"caller": caller_id})]),
)
agent = VoiceAgent(model, executor, session.speaker, speed=speed)
await session.run(agent)  # warm-up starts with the call
```

Run the whole thing offline: `python examples/voxcat_speed.py`.

## Start the reply before the turn is over

After `early_ms` of quiet, `VoiceSession` already starts the final transcription while the
turn detector decides. With `Speed`, that early transcript also starts the reply. Its words
are held back. When the turn ends with the same words and nothing was said in between, the
reply is kept and plays at once; if the caller goes on, it is dropped.

```python
--8<-- "examples/voxcat_speed.py:speculate"
```

Its model calls bill under the ledger stage `voice_speculation`, and each one emits a
`custom.voice.speculation` event (`started`, `adopted`, `dropped`). It saves the time the
turn detector waits after the transcript: much with a silence end-of-turn, little with a
fast semantic one.

## Run read-only tools early, and nothing else

The gate sits in the executor, before guards, budgets and approval. While the reply is
speculative, a tool call runs only if the tool is named in `SpeculativeTools`, declared
`effects="read_only"` and has no `confirm`. It is checked when the agent is built and again
on every call. Any other call waits until the turn is final, and never runs if the caller
goes on. The model's words cannot change this.

```python
--8<-- "examples/voxcat_speed.py:tools"
```

- `max_per_turn` (2) and `max_per_minute` (30) cap early calls; past them, calls wait.
- Each early call is one unit of the ledger item `voxcat:speculative_tool`; price it like
  any metered item. Each emits `custom.voice.speculative_tool` with the tool name.
- Give these tools `cache_ttl`: a dropped reply's call finishes into the cache, and the
  next reply shares it while it runs (in-flight sharing), so the tool runs once.

## Acknowledge a slow reply, never a fast one

The agent learns how long its replies take to start. When a turn ends on real speech and
the reply is expected later than `fast_ms`, a line plays at once, before the transcript
is back. Otherwise it waits `fast_ms` and plays only if the reply has not started. A kept
speculation or a cached answer gets none.

```python
--8<-- "examples/voxcat_speed.py:acks"
```

Lines play only after they are rendered at call start, so an acknowledgement is never
slower than the reply it covers. A speaker with no `warm()` (your own audio stack) speaks
them its own way.

## Replay answers to repeated questions

`AnswerCache` matches a question word for word (case and punctuation ignored), per tenant,
for `ttl_s`. The agent stores a reply only when it opened the call and used no tool, so it
depends on the question and the instructions alone. Questions under `min_words` (3) are
never stored. `put` seeds known answers. `answers.voice(tts)` keeps each stored reply's
audio and replays it with no synthesis; warm-up renders seeded answers.

```python
--8<-- "examples/voxcat_speed.py:answers"
```

Use one tenant per agent persona: two agents with different instructions should not share.

## Warm up at call start

`VoiceSession.run` calls `agent.warm_up()` as the call starts. It calls `warm()` on the
STT, TTS, model and each of `connections` (or calls a plain function), renders the
acknowledgements, `lines` and cached answers, and runs the prefetch `tools` through the
executor, all at once. A failed step is logged and listed in the `WarmReport`; the call goes
on. Prefetch tools follow the same rule as early tools: read-only, no `confirm`.

## Read the stage timings

Each `TurnTiming` in `session.timings` has the stages of its turn, and the
`custom.voice_turn` event carries them in `stages_ms`, in ms after the caller's last speech:
`early`, `ended`, `transcribed`, `speculated`, `replied`, `acked` and `first_audio`, plus
`adopted` and `answer_cached`.

## Measured

`python scripts/voxcat_speed.py` runs the real pipeline with fake STT, LLM and TTS on a
virtual clock, with approximate vendor-stated delays: STT final words 100 ms, first token
350 ms (900 ms for the slow model) then 40 tokens a second, first TTS audio 90 ms, a lookup
tool 300 ms. Flags change them. Times are ms after the caller's last speech.

| Scenario | 800 ms silence | [EagerTurns](voxcat-own.md) (600 ms, 250 ms when done) | 160 ms smart |
|---|---|---|---|
| plain question, no tricks | 1395 | 845 | 755 |
| plain question, speculative reply | 895 | 695 | 655 |
| tool question, no tricks | 2045 | 1495 | 1405 |
| tool question, speculative reply, tool waits | 1695 | 1345 | 1305 |
| tool question, speculative reply and tool | 1445 | 1345 | 1305 |
| slow model: first sound with an acknowledgement | 805 | 255 | 165 |
| repeated question: answer cache | 805 | 255 | 165 |

With an 800 ms silence end-of-turn, the speculative reply saves 500 ms and early tools 250
ms more on a tool question. `EagerTurns` needs no model and no provider feature and lands
close to a 160 ms semantic end-of-turn. The pipeline's own CPU time per 20 ms frame was
about 25 µs at 16 kHz and 0.2 ms for 8 kHz phone audio (most of it resampling) on a laptop.
Against LiveKit and Pipecat on the same parts: [VoxCat benchmark](voxcat-benchmark.md).
