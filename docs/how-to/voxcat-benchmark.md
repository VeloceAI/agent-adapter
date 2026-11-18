---
title: "VoxCat benchmark: first sound, cut-ins and barge-in against LiveKit Agents and Pipecat"
description: "VoxCat, LiveKit Agents and Pipecat run on the same fake speech-to-text, model and text-to-speech in real time. VoxCat with plain parts starts speaking about 0.4 s sooner than the others with a Flux-like STT, and stops within one frame when the caller cuts in."
---

# VoxCat benchmark

`scripts/voxcat_vs.py` runs three voice stacks on the same caller and the same fake parts, in
real time, each with its own turn taking, reply and interruption code:

- VoxCat: `VoiceSession` with `EagerTurns` and `Speed`;
- LiveKit Agents 1.8: `AgentSession`;
- Pipecat 1.16: a standard voice pipeline.

There are no keys, no network and no models; only the fakes' timing is simulated.

## The parts every stack gets

| Part | Timing (vendor-stated, changeable with flags) |
|---|---|
| VAD | one energy rule for all |
| streaming STT | interim words while the caller speaks; final words 100 ms after it is asked, or 300 ms after its own endpoint |
| model | first token at 350 ms, then 40 tokens a second |
| TTS | first audio 90 ms after each request |
| Flux-like STT (them only) | an eager end of turn 150 ms after speech, the end of turn at 260 ms, as Deepgram Flux sends them |

## The setups

- **defaults:** each stack as it ships.
  - LiveKit: VAD turns, a 0.5 s endpoint delay, preemptive generation on.
  - Pipecat: a 0.2 s VAD stop and its Smart Turn model.
  - VoxCat: `EagerTurns` on a 600 ms silence, the early pass at 100 ms.
- **same 500 ms wait:** every end-of-turn silence is 500 ms.
- **them: Flux-like STT:** LiveKit and Pipecat use the STT's turn events the way their Flux
  integrations do (LiveKit `turn_detection="stt"` with preemptive generation; Pipecat's
  eager turn strategies). VoxCat keeps plain parts.

The column **voxcat-sentences** is VoxCat speaking sentence by sentence, as the other two do.
It shows how much comes from the end of turn alone.

## Results

20 runs each, on a Windows laptop. Times are ms after the caller's last speech, p50 / p95.

**First sound, plain question**

| Setup | VoxCat | VoxCat, sentences | LiveKit | Pipecat |
|---|---|---|---|---|
| defaults | **592** / 624 | 895 / 939 | 1161 / 1176 | 1180 / 1220 |
| same 500 ms wait | **622** / 650 | 898 / 939 | 1160 / 1190 | 1468 / 1542 |
| them: Flux-like STT | **594** / 653 | 895 / 947 | 987 / 1026 | 973 / 1006 |

**The caller pauses 450 ms mid-sentence:** no stack cut in, and every stack still answered.

| Setup | VoxCat | LiveKit | Pipecat |
|---|---|---|---|
| defaults: first sound, p50 | 647 | 1358 | 1305 |
| model tokens per call | 19 | 34 | 17 |

VoxCat starts one reply in the pause and drops it after its first tokens. LiveKit writes a
whole reply it never plays.

**The caller cuts in while the agent talks:** time until the agent's audio stops, p50.

| Setup | VoxCat | LiveKit | Pipecat |
|---|---|---|---|
| defaults | **31** | 493 | 341 |
| them: Flux-like STT | **31** | 470 | 217 |

VoxCat pauses its audio within a frame and resumes it if the sound was a cough. LiveKit waits
its 0.5 s `min_interruption_duration`. Pipecat waits for its VAD to start.

## Where the time goes

At about 590 ms, VoxCat is close to the floor these parts allow:

| Stage | Time |
|---|---|
| early pass after 100 ms of quiet | 100 ms |
| first token (simulated) | 350 ms |
| first audio (simulated) | 90 ms |
| playout and timers | about 50 ms |

The others wait for the end of the turn before they commit, and for the whole first sentence
before they speak.

- **Sentence by sentence:** VoxCat is still about 90 ms ahead of the Flux-like setups, and
  265 ms ahead of their defaults.
- **The opener ("Sure,"):** speaking it at once saves about 300 ms more.

## Run it

```bash
python -m venv bench
bench/Scripts/pip install -e . "livekit-agents>=1.8,<2" "pipecat-ai>=1.13,<2"
cd scripts
../bench/Scripts/python voxcat_vs.py --runs 20 --json results.json
../bench/Scripts/python voxcat_vs.py --llm-first-ms 200 --tts-first-ms 75 --frameworks voxcat,livekit
```

On macOS or Linux the path is `bench/bin/`. Each stack's harness is its own file
(`voxcat_vs_livekit.py`, `voxcat_vs_pipecat.py`, `voxcat_vs_voxcat.py`). Each trial's `notes`
in the JSON record its settings.

## Read it with care

- **Fakes, not models.** Real STT, models and networks add their own time and jitter, the
  same for every stack. To measure your own providers, put them in the harnesses.
- **Pipecat's Smart Turn runs on tones, which it cannot read.** Its inference still runs, and
  its time counts, but its verdict is replaced with the truth. That is a best case for Pipecat.
- **LiveKit's turn model is skipped.** It needs LiveKit's hosted inference, so LiveKit runs
  VAD turns. Its echo warm-up is off, so it can be interrupted at once.
- **Timers on Windows round up to about 15 ms.** This slows every stack's fake token stream
  the same way.
- **VoxCat's word rules are tested on scripted text.** False cut-ins on real speech need real
  recordings (`scripts/turn_bench.py`).
- **The single-pipeline view:** [Answer sooner](voxcat-speed.md) measures each speed trick on
  a virtual clock.
