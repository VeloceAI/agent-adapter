---
title: "VoxCat on its own: its own end of turn, early words and one config file"
description: "Make VoxCat fast with any speech-to-text and text-to-speech, hosted or open: EagerTurns ends a turn sooner when the words look finished, the reply starts on the STT's interim words, the first words play before the sentence ends, and VoiceConfig sets up a whole call from one YAML file."
---

# VoxCat on its own

VoxCat does not need a provider's speed features, such as Deepgram Flux's early end of turn or
Cartesia's continuations. These four parts work with any STT and TTS, including the
[open models](voxcat-open-models.md):

| Part | What it does |
|---|---|
| `EagerTurns` | ends the turn 250 ms after speech when the words look finished, and holds it open when they trail off |
| interim words | the reply starts on the STT's interim words as soon as the caller pauses |
| `Speed(first_words=4)` | the first words play before the first sentence is finished |
| `VoiceConfig` | a whole call from one YAML file: `agent-adapter voice serve voxcat.yaml` |

Run it all offline: `python examples/voxcat_fast.py`. The numbers against LiveKit and
Pipecat on the same parts are in [VoxCat benchmark](voxcat-benchmark.md).

## End the turn when the words are finished

`EagerTurns` wraps `SilenceTurns`, `SileroTurns`, or a `SmartTurn` around one. `VoiceSession`
hands it the early transcript; `looks_done` reads it in a few microseconds, with no model:

- **Finished:** it ends with `.`, `?` or `!`, or it is a short answer ("yes", "no thanks").
  The turn ends `done_ms` (250) after speech.
- **Trailing off:** it ends with "and", "the", "um", a comma, or digits read out in groups.
  The turn waits up to `hold_ms` (1500), so the caller is not cut off.
- **Unsure:** the detector's own wait (its `end_ms`).

```python
--8<-- "examples/voxcat_fast.py:eager"
```

It learns each caller during the call:

- If they often pause mid-turn (say for 400 ms), a finished-looking turn waits a little
  longer than that pause.
- If it ended a turn and the caller went on within a second, `done_ms` grows by 100 ms.

It needs punctuation, which most streaming STTs send (Deepgram `smart_format`,
AssemblyAI, Whisper). With an STT that sends none, only the trailing-word rule applies.

## Start the reply on the interim words

After `early_ms` of quiet (100 is a good start), `VoiceSession` asks the STT for its final
words. If the stream also has a `partial` attribute, the reply starts on it at once. It is
kept when the final words match, and redone when they don't.

- **Have it:** `DeepgramSTT` (interim results) and `SonioxSTT`.
- **Your own STT:** add `partial` to its stream.

A pause mid-sentence starts a reply that is then dropped: it costs one model call, cut off
at its first tokens (the benchmark counts them).

## Speak the first words sooner

Without it, speech waits for the first full sentence. With `Speed(first_words=4)` (the
default):

- **First line:** it goes out at its first clause break (`,` `;` `:` or a dash) once it has
  4 words, or after 8 words with no break.
- **A short opener:** "Sure,", "Okay." or "Hi Ana," goes out alone at once, and the next
  line is eager too.
- **After that:** sentence by sentence.

`first_words=None` turns it off.

```python
--8<-- "examples/voxcat_fast.py:first-words"
```

## A whole call from one file

```yaml
# voxcat.yaml
stt: deepgram                        # or {use: "mypkg.speech:MySTT", any: options}
tts: {use: cartesia, voice: "your-voice-id"}
model: {use: openai, model: gpt-4.1-mini}
instructions: You book tables. Keep answers short.
tools: mypkg.tools:registry          # optional
turns: {end_ms: 600, eager: true}    # eager: EagerTurns
early_ms: 100
speed: {first_words: 4, acks: ["One moment."]}   # speed: false turns it off
```

```bash
agent-adapter voice check voxcat.yaml   # print what it sets up
agent-adapter voice serve voxcat.yaml   # talk to it on this computer's mic and speakers
```

In code, `VoiceConfig.load(path_or_dict).build(transport)` returns the session and the
agent, for any transport (phone, WebRTC, WebSocket):

```python
--8<-- "examples/voxcat_fast.py:config"
```

- **Short names:**
  - STT: `deepgram`, `deepgram-flux`, `openai`, `cartesia`, `elevenlabs`, `assemblyai`,
    `soniox`, `moonshine`, `whisper`, `parakeet`, `kyutai`;
  - TTS: `cartesia`, `elevenlabs`, `openai`, `fish`, `rime`, `moonshine`, `kyutai`,
    `chatterbox`, `orpheus`;
  - models: `openai`, `anthropic`, `gemini`.

  Anything else is `package.module:Name`.
- **Keys:** they stay in the environment (`DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, ...). An
  option named like a key or token is refused.
- **Approvals:** `confirm=True` tools need an executor with an approval port:
  `build(transport, executor=...)`. Without one they are refused.
- **YAML:** needs `pyyaml`; a `.json` file needs nothing.

## What it cannot do

- It cannot make a model itself faster. It removes the waits around it.
- The word rules are checked on scripted cases. To score them on real speech, use
  `scripts/turn_bench.py` and the Smart Turn test set.
