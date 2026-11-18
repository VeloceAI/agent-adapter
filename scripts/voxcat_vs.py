"""voxcat-vs: VoxCat against LiveKit Agents and Pipecat, on the same fake parts and real time.

    python -m venv bench
    bench/Scripts/pip install -e . "livekit-agents>=1.8,<2" "pipecat-ai>=1.13,<2"
    cd scripts && ../bench/Scripts/python voxcat_vs.py --runs 10
    python voxcat_vs.py --runs 50 --json results.json --frameworks voxcat,livekit

Every framework gets the same caller, VAD rule, streaming STT, model and TTS timings
(voxcat_vs_common.py); each runs its own turn taking, reply and interruption code. Setups:
- defaults: each framework as it ships (VoxCat: EagerTurns, Speed);
- same 500 ms wait: every end-of-turn silence set to 500 ms;
- them: Flux-like STT: LiveKit and Pipecat get an STT that says when the turn ends (and
  an eager end, like Deepgram Flux) and use it; VoxCat keeps plain parts.

Results are ms after the caller's last speech. No keys, no network, no models. Trials run in
`--jobs` processes so each one keeps the real clock to itself.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voxcat_vs_common import SCENARIOS, SETUPS, Delays, Trial, summary

FRAMEWORKS = ("voxcat", "voxcat-sentences", "livekit", "pipecat")
# voxcat-sentences: VoxCat speaking sentence by sentence, as LiveKit and Pipecat do


def run_group(framework: str, setup_i: int, scenario_i: int, runs: int, delays: dict) -> list[dict]:
    """One process: `runs` trials of one framework, setup and scenario, one after another."""
    import logging

    logging.disable(logging.WARNING)
    options = {"first_words": None} if framework == "voxcat-sentences" else {}
    try:
        harness = importlib.import_module(f"voxcat_vs_{framework.split('-')[0]}")
    except ImportError as exc:
        trial = Trial(framework, SETUPS[setup_i].name, SCENARIOS[scenario_i].name, error=repr(exc))
        return [asdict(trial)]

    async def trials() -> list[dict]:
        done = []
        for _ in range(runs):
            setup, scenario = SETUPS[setup_i], SCENARIOS[scenario_i]
            trial = await harness.run_trial(setup, scenario, Delays(**delays), **options)
            done.append({**asdict(trial), "framework": framework})
        return done

    return asyncio.run(trials())


def cell(stats: dict, *keys: str) -> str:
    values = [stats.get(key) for key in keys]
    if stats.get("errors") == stats.get("runs"):
        return "error"
    return " / ".join("-" if v is None else str(v) for v in values)


def table(results: dict[tuple[str, str, str], dict], frameworks: list[str]) -> str:
    lines = []
    views = (
        ("plain", "first sound (p50 / p95 ms)", ("first_p50", "first_p95")),
        (
            "mid-turn pause",
            "cut-ins / first sound p50 / model calls",
            ("cut_ins", "first_p50", "llm_calls"),
        ),
        (
            "barge-in",
            "stops talking after a cut-in (p50 ms) / first sound p50",
            ("stop_p50", "first_p50"),
        ),
    )
    for scenario, title, keys in views:
        lines += [
            "",
            f"{scenario}: {title}",
            "  " + "setup".ljust(24) + "".join(f.ljust(22) for f in frameworks),
        ]
        for setup in SETUPS:
            row = [cell(results.get((f, setup.name, scenario), {}), *keys) for f in frameworks]
            lines.append("  " + setup.name.ljust(24) + "".join(c.ljust(22) for c in row))
    return "\n".join(lines)  # CPU per frame is in --json; the fake parts dominate it


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--frameworks", default=",".join(FRAMEWORKS))
    parser.add_argument("--json", type=Path)
    for name, value in asdict(Delays()).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value)
    args = parser.parse_args()
    frameworks = [f for f in args.frameworks.split(",") if f]
    delays = {name: getattr(args, name) for name in asdict(Delays())}
    groups = [
        (f, si, ci) for f in frameworks for si in range(len(SETUPS)) for ci in range(len(SCENARIOS))
    ]
    results: dict[tuple[str, str, str], dict] = {}
    trials: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_group, f, si, ci, args.runs, delays): (f, si, ci)
            for f, si, ci in groups
        }
        for future, (f, si, ci) in futures.items():
            done = future.result()
            trials += done
            stats = summary([replace(Trial(f, "", ""), **d) for d in done])
            results[(f, SETUPS[si].name, SCENARIOS[ci].name)] = stats
            errors = {d["error"] for d in done if d["error"]}
            if errors:
                print(f"{f} / {SETUPS[si].name} / {SCENARIOS[ci].name}: {sorted(errors)[0][:200]}")
    print(f"voxcat-vs: {args.runs} runs each, delays {delays}")
    print(table(results, frameworks))
    if args.json:
        rows = [
            {"framework": k[0], "setup": k[1], "scenario": k[2], **v} for k, v in results.items()
        ]
        args.json.write_text(
            json.dumps({"delays": delays, "summary": rows, "trials": trials}, indent=1)
        )


if __name__ == "__main__":
    main()
