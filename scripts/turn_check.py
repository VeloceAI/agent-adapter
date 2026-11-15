"""turn-check: run the turn detectors side by side on WAV files or a live microphone.

    python scripts/turn_check.py --wav hello.wav other.wav
    python scripts/turn_check.py --mic 15

Needs `agent-adapter[vad]` (plus `sounddevice` for --mic) and the two model files: pass
--silero and --smart-turn, or install pipecat-ai, whose bundled copies are used by default.
Microphone audio stays in memory and is never written to disk.
"""

from __future__ import annotations

import argparse
import queue
import time
import wave
from pathlib import Path
from typing import Any

from agent_adapter.voice.silero import SileroTurns
from agent_adapter.voice.smart_turn import RATE, SmartTurn
from agent_adapter.voice.turns import SilenceTurns, TurnDetector

FRAME = 320  # 20 ms at 16 kHz


def _bundled() -> tuple[Path | None, Path | None]:
    try:
        import pipecat
    except ImportError:
        return None, None
    base = Path(pipecat.__file__).parent
    smart = sorted((base / "audio/turn/smart_turn/data").glob("smart-turn-v3*cpu*.onnx"))
    return base / "audio/vad/data/silero_vad.onnx", (smart[-1] if smart else None)


def detectors(silero: Path, smart: Path) -> dict[str, TurnDetector]:
    return {
        "loudness": SilenceTurns(),
        "silero": SileroTurns(silero),
        "silero+smart-turn": SmartTurn(smart, SileroTurns(silero)),
    }


def run_frames(frames: Any, found: dict[str, TurnDetector], live: bool = False) -> None:
    started = time.perf_counter()
    for n, frame in enumerate(frames):
        at = n * FRAME / RATE
        for name, detector in found.items():
            for event in detector.feed(frame):
                extra = ""
                if isinstance(detector, SmartTurn) and event == "ended":
                    extra = f" (model {detector.last_probability or 0:.2f})"
                lag = f", {time.perf_counter() - started - at:+.2f}s lag" if live else ""
                print(f"  {at:6.2f}s  {name:<18} {event}{extra}{lag}")


def wav_frames(path: Path) -> Any:
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getsampwidth(), w.getnchannels()) != (RATE, 2, 1):
            raise SystemExit(f"{path}: needs 16 kHz, 16-bit, mono")
        data = w.readframes(w.getnframes()) + b"\x00\x00" * RATE * 2  # 2 s of quiet after
    return (data[i : i + FRAME * 2] for i in range(0, len(data), FRAME * 2))


def mic_frames(seconds: float) -> Any:
    import sounddevice as sd

    frames: queue.Queue[bytes] = queue.Queue()

    def heard(data: Any, _frames: int, _time: Any, status: Any) -> None:
        frames.put(bytes(data))

    device = sd.query_devices(kind="input")
    print(f"listening on {device['name']} for {seconds:g}s; speak, pause, speak again")
    with sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME,
                           callback=heard):  # fmt: skip
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            yield frames.get()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav", nargs="*", type=Path, default=[])
    parser.add_argument("--mic", type=float, help="listen to the microphone for N seconds")
    parser.add_argument("--silero", type=Path)
    parser.add_argument("--smart-turn", type=Path)
    args = parser.parse_args()
    silero, smart = _bundled()
    silero, smart = args.silero or silero, args.smart_turn or smart
    if silero is None or smart is None:
        raise SystemExit("pass --silero and --smart-turn, or install pipecat-ai")
    for path in args.wav:
        print(path.name)
        run_frames(wav_frames(path), detectors(silero, smart))
    if args.mic:
        run_frames(mic_frames(args.mic), detectors(silero, smart), live=True)


if __name__ == "__main__":
    main()
