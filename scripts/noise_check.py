"""Does `NoiseGate` help speech-to-text in noise? Word error with and without it, measured.

Kokoro voices (Moonshine TTS) read sentences; pink noise or babble (other Kokoro voices) is
mixed in at 0, 5 and 10 dB; Moonshine transcribes each clip with and without the filter.
Needs the `voice-local` and `vad` extras. Usage: python scripts/noise_check.py
"""

from __future__ import annotations

import asyncio
import re

import numpy as np

from agent_adapter.voice.denoise import NoiseGate
from agent_adapter.voice.moonshine import MoonshineSTT, MoonshineTTS
from agent_adapter.voice.pcm import Resampler

SENTENCES = [
    "I would like to book two tickets for the jazz concert on Friday night.",
    "Please cancel my order and refund the money to my card.",
    "What time does the pharmacy on Main Street open tomorrow morning?",
    "My account number ends in four seven two and I cannot log in.",
    "Can you move my appointment from Tuesday to Thursday afternoon?",
    "The package was delivered to the wrong address last week.",
]
BABBLE = [
    "The weather has been lovely this week and the garden is full of flowers.",
    "We should meet for lunch near the station after the meeting finishes.",
    "He said the train was late again because of works on the line.",
]
RATE = 16_000


async def speak(tts: MoonshineTTS, text: str) -> np.ndarray:
    chunks = [c async for c in tts.synthesize(text)]
    pcm = Resampler(tts.sample_rate, RATE).feed(b"".join(chunks))
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64)


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.normal(size=n))
    spectrum /= np.sqrt(np.maximum(np.arange(len(spectrum)), 1))
    return np.fft.irfft(spectrum, n=n)


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> bytes:
    lead = int(RATE * 1.0)  # a second of noise first, as on a real line
    noise = np.resize(noise, len(speech) + lead)
    padded = np.concatenate([np.zeros(lead), speech])
    gain = np.sqrt(np.mean(speech**2) / (np.mean(noise**2) * 10 ** (snr_db / 10)))
    out = padded + gain * noise
    out *= min(1.0, 30_000 / np.max(np.abs(out)))
    return np.round(out).astype("<i2").tobytes()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def filtered(pcm: bytes) -> bytes:
    gate = NoiseGate()
    return b"".join(gate(pcm[i : i + 640]) for i in range(0, len(pcm), 640))


async def main() -> None:
    rng = np.random.default_rng(0)
    tts, stt = MoonshineTTS(), MoonshineSTT()
    speech = [await speak(tts, s) for s in SENTENCES]
    others = [
        await speak(MoonshineTTS(v), b)
        for v, b in zip(
            ("kokoro_am_adam", "kokoro_af_heart", "kokoro_am_adam"), BABBLE, strict=True
        )
    ]
    babble = sum(np.resize(o, max(len(x) for x in others)) for o in others)
    print("noise    SNR   WER raw   WER filtered")
    for name in ("pink", "babble"):
        for snr in (0, 5, 10):
            raw = clean = 0.0
            for text, clip in zip(SENTENCES, speech, strict=True):
                noise = pink(len(clip) * 2, rng) if name == "pink" else babble
                pcm = mix(clip, noise, snr)
                raw += wer(text, await stt.transcribe(pcm, RATE))
                clean += wer(text, await stt.transcribe(filtered(pcm), RATE))
            n = len(SENTENCES)
            print(f"{name:7} {snr:3} dB  {raw / n:7.1%}   {clean / n:7.1%}")


if __name__ == "__main__":
    asyncio.run(main())
