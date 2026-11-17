---
title: "Voice call features: idle prompts, voicemail, recording, hang-up and more"
description: "Idle prompts, lines the caller cannot cut, an end-call tool, typed input, call recording, a thinking sound, voicemail detection and a noise filter for VoiceSession, all offline."
---

# Voice call features

Everything on this page plugs into [`VoiceSession`](voice-pipeline.md) and runs offline,
with no account or key. Most features are a `CallObserver`, passed in `observers=`:

```python
from agent_adapter.voice import (
    CallRecorder, IdlePolicy, NoiseGate, ThinkingSound, VoicemailDetector, VoiceSession,
    end_call_tool,
)

session = VoiceSession(
    transport, stt=stt, tts=tts, turns=turns,
    filters=[NoiseGate()],  # the caller's audio, before anything else hears it
    observers=[
        IdlePolicy(after_s=8),
        CallRecorder("calls/today.wav"),
        ThinkingSound.ticks(),
        VoicemailDetector(message="Hi, this is the clinic. Please call us back."),
    ],
)
registry.add(end_call_tool(session))  # the agent may hang up after its goodbye
agent = VoiceAgent(model, executor, session.speaker, mute_during_tools=True)
```

| Feature | How | What it does |
|---|---|---|
| Greeting or notice | `agent.say(text)` | speaks outside a model turn; the history keeps what was heard |
| A line the caller cannot cut | `agent.say(text, interruptible=False)` | caller speech during it is dropped, not queued |
| No cut-ins while a tool runs | `VoiceAgent(mute_during_tools=True)` | speech while a tool runs is dropped |
| "uh-huh" never cuts | built in (`backchannels=`) | a backchannel while the agent talks resumes playback |
| Silent caller | `IdlePolicy(after_s=8, hang_up_after=2)` | asks "Are you still there?", then says goodbye and hangs up |
| The agent hangs up | `end_call_tool(session)` | ends the call once the goodbye has played |
| Hang up from your code | `await session.hang_up(after_speech=True)` | uses the transport's own hang-up where it has one |
| Typed input | `session.send_text(text)` | a chat box beside the call; handled like a spoken turn |
| Recording | `CallRecorder(path)` | stereo WAV, caller left, agent right, lined up in time |
| Thinking sound | `ThinkingSound.ticks()` or `.from_wav(path)` | plays quietly while the agent thinks; stops for speech |
| Voicemail | `VoicemailDetector(message=...)` | waits for the beep, leaves the message, hangs up |
| Noise filter | `filters=[NoiseGate()]` | turns steady background noise down (numpy) |

## Voicemail on outbound calls

`VoicemailDetector` uses two offline signals in the first `window_s` (20 s) of a call:

- the greeting's words: "leave a message", "not available" and other `PHRASES`, or your
  own `classify(text)` (a model call, say);
- the beep: a steady tone of 400 to 2,500 Hz lasting 160 ms.

Once it hears a greeting, the machine's words never reach the agent. With `message=`, the
agent waits for the beep (or 4 s of quiet), speaks the message without cut-ins and hangs up.
With `on_detect(session, reason)` you decide; `reason` is `"greeting"` or `"beep"`.

Checked with real speech, not a person: two Kokoro voices read 3 greetings and 4 replies a
person might give. Moonshine's transcripts matched all 6 greetings and none of the 8 replies.
The longest steady tone found in any speech was 0 ms, and test beeps from 440 to 2,000 Hz
were all found. A real line adds noise and codecs: test with your carrier before relying on
it.

## The noise filter, measured

`NoiseGate` learns the steady noise in each frequency band and turns it down by up to 12 dB.
`scripts/noise_check.py` mixes noise into six Kokoro sentences and compares Moonshine's
word error with and without the filter (lower is better):

| Noise | Signal to noise | Without | With `NoiseGate` |
|---|---|---|---|
| Pink (hiss, fans) | 0 dB | 9.4% | 10.0% |
| Pink | 5 dB | 5.1% | 3.8% |
| Pink | 10 dB | 3.8% | 3.8% |
| Crowd babble (three voices) | 0 dB | 114.9% | **71.6%** |
| Crowd babble | 5 dB | 28.3% | **10.4%** |
| Crowd babble | 10 dB | 5.0% | 5.5% |

It helps most where Moonshine struggles, a loud crowd behind the caller, and changes little
elsewhere: Moonshine already copes with hiss. Above 100% means Moonshine wrote words that
were never said. Six sentences is a small test, and a crowd that changes all the time is
harder than ours, which repeats: try it on your own lines before turning it on.

Any function `pcm -> pcm` fits `filters=`, so a stronger denoiser plugs in the same way.

## Recording

`CallRecorder` keeps the call in memory (up to `max_minutes`, 60) and writes the WAV when the
call ends. Each side is placed at the time it was heard. Tell callers they are recorded
where the law asks you to; the file holds their voice, so store it like other personal data.

## Write your own

```python
from agent_adapter.voice import CallObserver


class Transcript(CallObserver):
    def __init__(self) -> None:
        self.lines: list[str] = []

    def caller_said(self, text: str) -> None:
        self.lines.append(text)  # return True instead to keep a line from the agent
```

`started(session)`, `caller_audio(pcm)`, `agent_audio(pcm)`, `caller_said(text)` and
`ended()` are all optional. `agent_audio` can run on the sound card's thread: keep it quick.
An observer that raises is logged and the call goes on.
