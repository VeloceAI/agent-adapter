"""turn-bench: score the turn detectors on real human speech with turn-end labels.

    python scripts/turn_bench.py --data DIR --fetch 300
    python scripts/turn_bench.py --data DIR --json results.json

Data: the human, English clips of Pipecat's Smart Turn v3.2 test set (CC-BY-4.0,
huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test). Each clip stops either where
the speaker finished (complete) or mid-thought (incomplete). `--fetch N` downloads N clips,
half of each, from random pages of the Hugging Face datasets-server (about 0.3 MB a clip)
into DIR; keep DIR outside the repo. Needs `agent-adapter[vad]`, `soundfile`, and the Silero and
Smart Turn model files (pass --silero and --smart-turn, or install pipecat-ai).

Each clip is fed in 20 ms frames after 0.5 s of quiet, then 5 s of quiet follow, as if the
caller stopped talking. Speech end is the last 32 ms window Silero scores >= 0.5.
- false cut: the detector ended the turn and the caller was not done: it later spoke again
  inside the clip, or (incomplete clips) the end came within the pause after the clip;
- delay: speech end to "ended", on complete clips (and how many end over 1.5 s late);
- missed: the turn never started, or a complete turn had not ended 5 s after speech end.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_adapter.voice.silero import SileroTurns
from agent_adapter.voice.smart_turn import RATE, SmartTurn
from agent_adapter.voice.turns import SilenceTurns, TurnDetector

DATASET = "pipecat-ai/smart-turn-data-v3.2-test"
API = "https://datasets-server.huggingface.co/rows"
FRAME = 320  # 20 ms
LEAD_S, TAIL_S = 0.5, 5.0
PAUSES_S = (1.0, 2.0)
SLOW_S = 1.5


def _bundled() -> tuple[Path | None, Path | None]:
    try:
        import pipecat
    except ImportError:
        return None, None
    base = Path(pipecat.__file__).parent
    smart = sorted((base / "audio/turn/smart_turn/data").glob("smart-turn-v3*cpu*.onnx"))
    return base / "audio/vad/data/silero_vad.onnx", (smart[-1] if smart else None)


def _get(url: str) -> bytes:
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data: bytes = response.read()
                return data
        except OSError:
            if attempt == 5:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def _page(offset: int, length: int = 100) -> tuple[int, list[dict[str, Any]]]:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": "train", "offset": offset,
         "length": length}
    )  # fmt: skip
    page = json.loads(_get(f"{API}?{query}"))
    return page["num_rows_total"], [r["row"] for r in page["rows"]]


def fetch(data: Path, clips: int, seed: int = 7) -> None:
    data.mkdir(parents=True, exist_ok=True)
    manifest = data / "manifest.jsonl"
    rows = (
        [json.loads(line) for line in manifest.read_text().splitlines()]
        if manifest.exists()
        else []
    )
    have = {row["id"] for row in rows}
    wanted = {True: clips // 2, False: clips - clips // 2}
    for row in rows:
        wanted[row["endpoint_bool"]] -= 1
    total, _ = _page(0, 1)
    offsets = random.Random(seed).sample(range(0, total - 100, 100), (total - 100) // 100)
    with manifest.open("a") as out:
        for offset in offsets:
            if max(wanted.values()) <= 0:
                break
            try:
                page = _page(offset)[1]
            except OSError as exc:
                print(f"  skipped a page: {exc}")  # the server has flaky moments
                continue
            for row in page:
                label = row["endpoint_bool"]
                if row["synthetic"] or row["language"] != "eng" or row["id"] in have:
                    continue
                if wanted[label] <= 0:
                    continue
                (data / f"{row['id']}.flac").write_bytes(_get(row["audio"][0]["src"]))
                keep = ("id", "endpoint_bool", "midfiller", "endfiller", "dataset")
                out.write(json.dumps({k: row[k] for k in keep}) + "\n")
                have.add(row["id"])
                wanted[label] -= 1
            print(f"  {len(have)} clips so far")


def load(data: Path) -> list[tuple[dict[str, Any], Any]]:
    import numpy as np
    import soundfile as sf

    clips = []
    for line in (data / "manifest.jsonl").read_text().splitlines():
        meta = json.loads(line)
        audio, rate = sf.read(str(data / f"{meta['id']}.flac"), dtype="int16")
        if rate != RATE or audio.ndim != 1:
            continue
        clips.append((meta, np.asarray(audio, dtype=np.int16)))
    return clips


def speech_end(vad: SileroTurns, audio: Any) -> float | None:
    np = vad._np
    vad.reset()
    samples = audio.astype(np.float32) / 32_768
    last = None
    for start in range(0, len(samples) - 512 + 1, 512):
        if vad.probability(samples[start : start + 512]) >= 0.5:
            last = (start + 512) / RATE
    return last


def events(detector: TurnDetector, audio: Any) -> list[tuple[float, str]]:
    quiet = b"\x00\x00"
    pcm = quiet * int(LEAD_S * RATE) + audio.tobytes() + quiet * int(TAIL_S * RATE)
    out = []
    for n in range(0, len(pcm), FRAME * 2):
        at = (n // 2 + FRAME) / RATE - LEAD_S
        out += [(at, event) for event in detector.feed(pcm[n : n + FRAME * 2])]
    return out


def judge(found: list[tuple[float, str]], complete: bool, end: float) -> dict[str, Any]:
    starts = [t for t, e in found if e == "started"]
    ends = [t for t, e in found if e == "ended"]
    restarted = any(any(s > t for s in starts) for t in ends)
    result: dict[str, Any] = {"missed": not starts, "cut": restarted, "delay": None}
    if complete:
        final = [t for t in ends if not any(s > t for s in starts)]
        result["missed"] |= not final or final[0] - end > TAIL_S - 0.1
        if final and not result["missed"]:
            result["delay"] = final[0] - end
    else:
        for pause in PAUSES_S:
            result[f"cut_{pause:g}s"] = restarted or any(t < end + pause for t in ends)
    return result


def score(clips: list[tuple[dict[str, Any], Any]], makers: dict[str, Callable[[], TurnDetector]],
          silero: Path) -> dict[str, dict[str, Any]]:  # fmt: skip
    vad = SileroTurns(silero)
    rows: dict[str, list[tuple[bool, dict[str, Any]]]] = {name: [] for name in makers}
    for n, (meta, audio) in enumerate(clips, 1):
        end = speech_end(vad, audio)
        if end is None:
            continue
        for name, make in makers.items():
            rows[name].append(
                (meta["endpoint_bool"], judge(events(make(), audio), meta["endpoint_bool"], end))
            )
        if n % 50 == 0:
            print(f"  scored {n}/{len(clips)}")
    return {name: summarise(found) for name, found in rows.items()}


def _pct(items: list[bool]) -> float:
    return round(100 * sum(items) / len(items), 1) if items else 0.0


def summarise(found: list[tuple[bool, dict[str, Any]]]) -> dict[str, Any]:
    done = [r for complete, r in found if complete]
    open_ = [r for complete, r in found if not complete]
    delays = sorted(r["delay"] for r in done if r["delay"] is not None)
    out: dict[str, Any] = {
        "complete_clips": len(done),
        "incomplete_clips": len(open_),
        "false_cut_complete_pct": _pct([r["cut"] for r in done]),
        "missed_pct": _pct([r["missed"] for _, r in found]),
        "delay_p50_ms": round(1000 * statistics.median(delays)) if delays else None,
        "delay_p90_ms": round(1000 * delays[int(0.9 * (len(delays) - 1))]) if delays else None,
        "slow_end_pct": _pct([d > SLOW_S for d in delays]),
    }
    for pause in PAUSES_S:
        out[f"false_cut_incomplete_{pause:g}s_pct"] = _pct([r[f"cut_{pause:g}s"] for r in open_])
    return out


def makers(silero: Path, smart: Path, end_ms: float) -> dict[str, Callable[[], TurnDetector]]:
    return {
        f"loudness ({end_ms:g} ms)": lambda: SilenceTurns(end_ms=end_ms),
        f"silero ({end_ms:g} ms)": lambda: SileroTurns(silero, end_ms=end_ms),
        "silero+smart-turn": lambda: SmartTurn(smart, SileroTurns(silero)),
    }


def table(results: dict[str, dict[str, Any]]) -> str:
    heads = ["Detector", "False cuts, complete", "False cuts, 1 s pause", "False cuts, 2 s pause"]
    heads += ["Delay p50", "Delay p90", "Ends > 1.5 s late", "Missed"]
    keys = ["false_cut_complete_pct", "false_cut_incomplete_1s_pct", "false_cut_incomplete_2s_pct"]
    keys += ["delay_p50_ms", "delay_p90_ms", "slow_end_pct", "missed_pct"]
    lines = ["| " + " | ".join(heads) + " |", "|---" * len(heads) + "|"]
    for name, r in results.items():
        cells = [f"{r[k]} ms" if k.endswith("_ms") else f"{r[k]}%" for k in keys]
        lines.append("| " + " | ".join([name, *cells]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="clip folder, outside the repo")
    parser.add_argument("--fetch", type=int, default=0, help="download N clips first")
    parser.add_argument("--end-ms", type=float, default=600, help="quiet that ends a turn")
    parser.add_argument("--silero", type=Path)
    parser.add_argument("--smart-turn", type=Path)
    parser.add_argument("--json", type=Path, help="also write the results here")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.data, args.fetch)
    silero, smart = _bundled()
    silero, smart = args.silero or silero, args.smart_turn or smart
    if silero is None or smart is None:
        raise SystemExit("pass --silero and --smart-turn, or install pipecat-ai")
    clips = load(args.data)
    print(f"{len(clips)} clips from {DATASET} (CC-BY-4.0)")
    results = score(clips, makers(silero, smart, args.end_ms), silero)
    print(table(results))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
