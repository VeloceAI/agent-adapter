---
title: "VoxCat quality checks: bad audio, unsure words, names and spoken numbers"
description: "Fast, model-free checks for a voice call: caller audio level, clipping and dropouts, read-back of unsure amounts before a confirmed tool, a known-names fix, spoken money and dates (rupees, lakh, crore) and glitches in the agent's own voice, each under a millisecond."
---

# Quality checks for a voice call

VoxCat, the voice pipeline in `agent_adapter.voice`, ships five quality checks in
`agent_adapter.voice.quality`. Each is plain Python with no model, key or download, and
costs well under a millisecond a frame or a sentence (a test holds them to that).

| Check | What it guards | Plugs in as |
|---|---|---|
| `InputCheck` | the caller's audio: level, clipping, noise floor, dropouts | a filter and an observer |
| `UnsureWords` | numbers, dates and names the speech-to-text was unsure of | `VoiceApproval(readback=...)` |
| `KnownNames` | names the speech-to-text misheard | `check_stt(stt, fixes=[...])` |
| `SpokenForm` | money, numbers, dates, times, short forms, a pronunciation dictionary | an observer |
| `OutputCheck` | clicks, clipping and dead air in the agent's own voice | an observer |

## All of them on one call

```python
from agent_adapter.voice import VoiceAgent, VoiceApproval, VoiceSession
from agent_adapter.voice.quality import (
    InputCheck, KnownNames, OutputCheck, SpokenForm, UnsureWords, check_stt,
)

unsure = UnsureWords()
stt = check_stt(my_stt, fixes=[KnownNames(["Rahul", "Acme Corp"])], unsure=unsure)
caller = InputCheck(gate_db=-55)
session = VoiceSession(
    transport, stt=stt, tts=tts, turns=turns,
    filters=[caller],
    observers=[stt, caller, SpokenForm(locale="en-IN"), OutputCheck()],
)
executor = Executor(registry, approval=VoiceApproval(readback=unsure.readback))
agent = VoiceAgent(model, executor, session.speaker)
```

`stt` goes in `observers=` too, so the words it notes land under the call's run id.

## Spoken form

The text-to-speech gets the words a person would say. Plain numbers read in lakh and crore
with `locale="en-IN"`, in millions with "en-US"; rupees always use lakh and crore.

--8<-- "examples/voxcat_quality.py:spoken"

```text
vox cat refunds four lakh fifty thousand rupees on eighteenth November twenty twenty-six at two thirty P M.
Doctor Rao paid one thousand two hundred forty-nine dollars and ninety-nine cents, approximately three point five percent more, on November eighteenth, twenty twenty-six.
```

It covers rupees (`₹`, `Rs`), dollars, euros and pounds, `k`, `M`, lakh and crore
after an amount, decimals, ranges ("10-20"), percentages, ordinals, ISO, slash and written
dates, times, phone numbers (digit by digit), units (km, kg, mins) and short forms (Dr.,
e.g., etc.). Add more money with `currencies={"AED": Currency("dirham", "dirhams", "fils",
"fils")}`. `pronounce` is case sensitive, so "SQL" and "sql" can differ. As an observer it
adds itself to `session.speaker.text_hooks`; `Player(text_hooks=[...])` does the same
without a session. The history keeps the agent's own words, not the spoken form.

## Fix misheard names

--8<-- "examples/voxcat_quality.py:names"

A word changes when it is close in spelling (`cutoff=0.8`) or sounds the same and is
close enough (`sound_cutoff=0.65`). Short words and common English words never change.
Exact matches only get the name's capitals. `last_fixes` lists the guesses.

## Read back unsure words before a confirmed tool

When the speech-to-text was unsure of an amount, date or name that a `confirm=True` tool
is about to use, the spoken approval checks it first. One yes answers both.

--8<-- "examples/voxcat_quality.py:unsure"

```text
Did you say 4,500? Shall I run refund with amount 4500? Say yes to confirm.
```

A word under `threshold` counts. Only the ones found in the tool's arguments are read
back (`match_arguments=False` reads all), each once. The words come from `check_stt`, or
from your code with `unsure.hear(words, run_id=agent.ctx.run_id)`. `check_stt` takes word
confidence from the STT, or its stream, setting `last_words` (a list of `Word`, or the
word dicts Deepgram, AssemblyAI and Soniox send). With no confidence every word counts as
sure, but a name `KnownNames` guessed is still read back.

--8<-- "examples/voxcat_quality.py:stt"

## Bad audio from the caller

`InputCheck` measures each 16 kHz frame. As a filter, frames under `gate_db` become
silence, so hiss stops waking the turn detector; it holds open `hold_ms` after speech.

After each turn it names the problems: `clipping`, `dropouts` (short gaps of exact silence
with speech on both sides), `too quiet` or `noisy` (a high noise floor). A turn with a
problem is kept from the agent, and the caller hears "Sorry, I didn't catch that clearly.
Could you say it again?" (`repeat_prompt`), at most `max_repeats` times in a row.
`ask_again=False` only counts.

## Glitches in the agent's voice

`OutputCheck` reads the agent's audio as it plays: a big jump between two samples is a
click, samples at full scale are clipping, and `silence_s` of near silence inside a line
is dead air from a broken voice. It runs on the playout thread, so it only counts:
`counts`, the last `issues`, and `on_issue(issue)` if you pass one.

--8<-- "examples/voxcat_quality.py:audio"

At the end of a call both audio checks send their counts to the trace as a
`custom.voice_quality` event, with `side` set to `caller` or `agent`. No words are in it.

## What runs off the hot path

Checks that need a model, such as translation quality or transcript accuracy against a
second speech-to-text, cost far more than a millisecond. Run them after the call, on the
recording (`CallRecorder`) and the trace, or in [evals](evaluate-your-agent.md).

## See also

- [A voice call with no framework](voice-pipeline.md): `VoiceSession`
- [Voice call features](voice-call-features.md): more observers
- [Safe tools on a call](voice-agents.md): spoken approval
- Full example: `examples/voxcat_quality.py`
