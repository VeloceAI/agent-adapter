---
title: "Open speech models for VoxCat: Whisper, Parakeet, Chatterbox, Orpheus, Kyutai"
description: "Run our VoxCat voice pipeline on open models you host: faster-whisper, NVIDIA Parakeet, Kyutai STT and TTS, Chatterbox, Orpheus, and Haru-4B or Moshi as an instant talker next to the agent."
---

# Use open speech models

VoxCat, our own voice pipeline (`agent_adapter.voice`), runs on open models you host as
well as on hosted APIs. Each adapter fits the same ports as the
[Moonshine models](voice-pipeline.md), so you swap one line. A library is imported only
when you build its adapter; if it is missing you get `MissingDependency` with the pip
command.

## Models and licences

| Model | Task | Class | Licence | Use | Extra | How tested |
|---|---|---|---|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper) | STT | `FasterWhisperSTT` | MIT code and weights | commercial | `voice-whisper` | fake `WhisperModel` from the README |
| [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) v3, v2 | STT | `ParakeetSTT` | weights CC-BY-4.0, NeMo Apache-2.0 | commercial, credit NVIDIA | `voice-parakeet` | fake `ASRModel` from the model card |
| [Kyutai STT](https://huggingface.co/kyutai/stt-1b-en_fr) 1B, 2.6B | streaming STT | `KyutaiSTT` | weights CC-BY-4.0, code MIT | commercial, credit Kyutai | `voice-kyutai` | fake `moshi` from `stt_from_file_pytorch.py` |
| [Chatterbox](https://github.com/resemble-ai/chatterbox) Turbo, multilingual | TTS | `ChatterboxTTS` | MIT code and weights | commercial; archived, watermarked | `voice-chatterbox` | fake models from the README |
| [Orpheus 3B](https://github.com/canopyai/Orpheus-TTS) | streaming TTS | `OrpheusTTS` | Apache-2.0 code and weights | commercial; Llama 3.2 base | `voice-orpheus` | fake `OrpheusModel` from the README |
| [Kyutai TTS 1.6B](https://huggingface.co/kyutai/tts-1.6b-en_fr) | streaming TTS | `KyutaiTTS` | weights CC-BY-4.0, code MIT | commercial, credit Kyutai | `voice-kyutai` | fake `TTSModel` from `tts_pytorch.py` |
| [Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B) | talker | `HaruTalker` | Apache-2.0 code and weights | research release | none (see below) | fake `DuplexSession` from the README |
| [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) | talker | `MoshiTalker` | weights CC-BY-4.0, code MIT | **research only** | `voice-kyutai` | fake `moshi` from its README |

Licences were read from each model's official page on 2026-11-18. Check them again before
you ship.

!!! warning "Research only"
    Moshi's model card says the model is for research only, so we list it as research
    only: do not put it in a product. Haru-4B is Apache-2.0, but higurashi releases it
    as a research base model; use it for short acknowledgements, never for facts.

Notes on the rest: CC-BY-4.0 lets you use the weights commercially if you credit the
author. Orpheus is a fine-tune of Llama 3.2, so read Meta's Llama 3.2 licence as well.
Chatterbox was retired in August 2026: the code stays public under MIT with no updates,
and every clip carries Resemble's PerTh watermark.

## Install

Each family has its own extra. All but `voice-whisper` pull PyTorch, so give each one its
own virtual environment.

```bash
pip install "agent-adapter[voice-whisper]"
```

- `voice-parakeet` needs Python 3.12 or newer (NeMo's floor).
- `voice-orpheus` runs on vLLM: a CUDA GPU on Linux.
- Haru is not on PyPI: clone `github.com/higurashi-ai/Haru`, then `pip install -e ./haru`.

Models download from Hugging Face on first use, not at install.

## Speech-to-text

```python
from agent_adapter.voice.kyutai import KyutaiSTT
from agent_adapter.voice.parakeet import ParakeetSTT
from agent_adapter.voice.whisper import FasterWhisperSTT

stt = FasterWhisperSTT("large-v3-turbo", device="cuda", compute_type="float16")
stt = ParakeetSTT("nvidia/parakeet-tdt-0.6b-v3", device="cuda")
stt = KyutaiSTT("kyutai/stt-1b-en_fr", device="cuda")  # transcribes while the caller talks
text = await stt.transcribe(pcm, 16_000)
```

Whisper and Parakeet transcribe the finished turn. `FasterWhisperSTT` decodes greedily
(`beam_size=1`) with a fixed language by default, the fast setting for short turns. Kyutai
STT decodes each 80 ms frame as it arrives, so when the turn ends only the model's delay
is left: pass it to `VoiceSession` as a streaming STT.

## Text-to-speech

```python
from agent_adapter.voice.chatterbox_tts import ChatterboxTTS
from agent_adapter.voice.kyutai import KyutaiTTS
from agent_adapter.voice.orpheus import OrpheusTTS

tts = ChatterboxTTS(variant="turbo", voice="agent_voice.wav")  # a clip you may clone
tts = OrpheusTTS(voice="tara")
tts = KyutaiTTS(voice="expresso/ex03-ex01_happy_001_channel1_334s.wav")
async for chunk in tts.synthesize("Your order ships today."):
    ...  # PCM16 at 24 kHz, as soon as each chunk is ready
```

Orpheus and Kyutai stream frame by frame. Chatterbox renders whole clips, so the adapter
splits a line at sentence ends and plays the first sentence while the next one renders.
When the caller cuts in, rendering stops at the next chunk.

## Keep models warm

A model loads once per process and is shared by every adapter built with the same
settings, so a new call does not reload it. Loading happens in a worker thread on first
use; call `warm()` at start-up to load it and run one tiny inference ahead of the first
caller:

```python
await stt.warm()
await tts.warm()
```

One loaded model runs one inference at a time. Build a second adapter on another device
(`device="cuda:1"`) to serve calls side by side.

## Add a talker next to the agent

A talker is a full-duplex model: it hears the caller and speaks at the same time. It fills
the gap while the agent (the thinker) calls tools: "mm-hm", "right", "one moment". It
never answers for the agent. The port is small:

- `Talker.open(rate)` starts a `TalkerStream`; `warm()` loads the model ahead of time.
- `stream.hear(pcm)` passes the caller's audio; it returns at once.
- `stream.cues()` yields each `Cue(audio, text)` the talker says; silence is left out.
- `backchannel(stream, play, may_speak=...)` plays cues only while `may_speak()` is true.

```python
import asyncio

from agent_adapter.voice.haru import HaruTalker
from agent_adapter.voice.talker import backchannel

talker = HaruTalker(system_prompt=card, voice="agent_voice.wav")  # MoshiTalker() also fits
await talker.warm()
stream = talker.open(16_000)
nods = asyncio.create_task(backchannel(stream, play, may_speak=lambda: not agent_speaking))
stream.hear(caller_frame)  # every frame the caller sends
```

`card` is Haru's character card (`haru.prompt.render_card`). The agent owns the turn: once
its answer starts, `may_speak` turns false and the talker's cues are dropped.

## Try it offline

[`examples/voxcat_open_models.py`](https://github.com/VeloceAI/agent-adapter/blob/main/examples/voxcat_open_models.py)
runs a call with stand-ins for each part, so it needs no model, GPU or download. On a GPU
machine with `voice-whisper` and `voice-kyutai`, `--real` swaps in faster-whisper, Kyutai
TTS and Moshi.

```bash
python examples/voxcat_open_models.py
```

## How these are tested

We do not download any model in CI. Each test puts a stand-in for the library in place,
shaped after the calls in its README or model card, and checks what the adapter sends and
how it turns the result into audio or text: resampling, sentence splitting, streaming,
barge-in, one load per process and the missing-library message.
