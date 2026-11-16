"""voice-race: our own pipeline against Pipecat and LiveKit, the same call on each.

    python scripts/voice_race.py --runs 5 --silero silero_vad.onnx --smart-turn smart.onnx \
        --livekit-server ./livekit-server --out race.jsonl

Every stack gets the same scripted caller, the same Silero + Smart Turn detectors, the same
Moonshine models and the same scripted model, and is timed from what the caller hears
(`Measure` in voice_live_common.py). Runs are silent: audio is pulled at a sound card's pace
but not played. Needs one environment with agent-adapter[voice-local,vad,pipecat,livekit]
and pipecat-ai[local]; LiveKit runs need a livekit-server binary (started here with --dev).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
STACKS = {
    "agent-adapter": "voice_live.py",
    "pipecat": "voice_live_pipecat.py",
    "livekit": "voice_live_livekit.py",
}
CUTS = (0.3, 0.8, 1.5, 2.5)


def run_once(stack: str, cut: float, args: argparse.Namespace) -> None:
    command = [sys.executable, str(HERE / STACKS[stack]), "--silent", "--cut", str(cut)]
    command += ["--silero", str(args.silero), "--smart-turn", str(args.smart_turn)]
    command += ["--json", str(args.out)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    if done.returncode not in (0, 1):
        print(f"  {stack} cut {cut}: exit {done.returncode}\n{done.stderr[-2000:]}")


def median(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if r.get(key) is not None]
    return f"{statistics.median(values):.0f}" if values else "-"


def table(rows: list[dict]) -> str:
    lines = [
        "| Stack | Runs | Charged once | First audio ms | Audio stops ms | Agent told ms | CPU s |",
        "|---|---|---|---|---|---|---|",
    ]
    for stack in STACKS:
        mine = [r for r in rows if r["stack"] == stack]
        if not mine:
            continue
        once = sum(r["reserved"] == r["charged"] == 1 for r in mine)
        lines.append(
            f"| {stack} | {len(mine)} | {once}/{len(mine)} | {median(mine, 'first_audio_ms')} | "
            f"{median(mine, 'audio_stop_ms')} | {median(mine, 'agent_cut_ms')} | "
            f"{median(mine, 'cpu_s')} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5, help="runs per stack and cut point")
    parser.add_argument("--silero", type=Path, required=True)
    parser.add_argument("--smart-turn", type=Path, required=True)
    parser.add_argument("--livekit-server", type=Path, help="run LiveKit too (dev mode)")
    parser.add_argument("--stacks", nargs="*", default=list(STACKS))
    parser.add_argument("--out", type=Path, default=Path("race.jsonl"))
    parser.add_argument("--table-only", action="store_true", help="print the table of --out")
    args = parser.parse_args()
    if not args.table_only:
        server = None
        stacks = [s for s in args.stacks if s != "livekit" or args.livekit_server]
        if "livekit" in stacks:
            command = [str(args.livekit_server), "--dev", "--bind", "127.0.0.1"]
            server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
        try:
            for n in range(args.runs):
                for cut in CUTS:
                    for stack in stacks:  # interleaved, so drift hits every stack alike
                        print(f"run {n + 1}/{args.runs}  cut {cut}s  {stack}", flush=True)
                        run_once(stack, cut, args)
        finally:
            if server is not None:
                server.terminate()
    rows = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines()]
    print(table(rows))
    for cut in CUTS:
        print(f"\ncut at {cut} s")
        print(table([r for r in rows if r["cut_s"] == cut]))


if __name__ == "__main__":
    main()
