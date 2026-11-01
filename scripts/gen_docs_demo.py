"""Write docs/assets/demo/sample-run.html: the run viewer on a recorded run of the fan-out example.

Run from the repo root: `python scripts/gen_docs_demo.py` (add `--check` to only compare).
The output is byte-stable: ids, times, the process id and versions are replaced by fixed
values, and the example's waits are skipped so its events always come in the same order.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import re
import runpy
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the checkout's agent_adapter, not an installed copy

from agent_adapter.viewer.assets import export_html  # noqa: E402  (needs the path above)
from agent_adapter.viewer.runs import RunFolder  # noqa: E402

EXAMPLE = ROOT / "examples" / "viewer_fan_out.py"
OUT = ROOT / "docs" / "assets" / "demo" / "sample-run.html"
BASE_TS = 1_790_000_000.0  # a fixed start time (2026-09-21 UTC)
STEP_S = 0.004  # time between two recorded events
IDS = re.compile(r"\b(run|apr|call|cmp)_[0-9a-f]{32}\b")
# Values that differ between recordings: times, durations, process ids.
VARYING = re.compile(r'"(ts|started|ended|pid|\w+_ms|\w+_s)":\s*-?[0-9.eE+-]+')


@contextlib.contextmanager
def _no_waits() -> Iterator[None]:
    """`asyncio.sleep(n)` yields once instead of waiting, so concurrent calls end in order."""
    real = asyncio.sleep

    async def instant(delay: float, result: Any = None) -> Any:
        return await real(0, result)

    asyncio.sleep = instant
    try:
        yield
    finally:
        asyncio.sleep = real


def record(folder: Path) -> Path:
    """Run the example into `folder`; the runs folder it wrote."""
    module = runpy.run_path(str(EXAMPLE), run_name="docs_demo")
    with _no_waits(), contextlib.redirect_stdout(io.StringIO()):
        return Path(asyncio.run(module["main"](folder)))


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _order(path: Path) -> tuple[bool, str]:
    """The flow run first (the docs link its id), then by content without ids, times or
    durations: the same order under any load."""
    text = "".join(_lines(path))
    return '"flow.started"' not in text, VARYING.sub("", IDS.sub("", text))


def normalise(runs: Path, out: Path) -> None:
    """Copy every run file with fixed ids, times, durations, process id and versions."""
    ids: dict[str, str] = {}

    def fixed(match: re.Match[str]) -> str:
        found = match.group(0)
        if found not in ids:
            ids[found] = f"{match.group(1)}_{len(ids) + 1:032x}"
        return ids[found]

    clock = 0
    for path in sorted(runs.glob("*.jsonl"), key=_order):
        records = [json.loads(IDS.sub(fixed, line)) for line in _lines(path) if line.strip()]
        started: dict[str, float] = {}
        first = BASE_TS + clock * STEP_S
        for record in records:
            if "ts" in record:
                record["ts"] = round(BASE_TS + clock * STEP_S, 6)
                clock += 1
            if record.get("kind") == "meta":
                record.update(pid=1, started=round(first, 6), python="3", agent_adapter="1")
            if record.get("kind") != "event":
                continue
            if record["type"].endswith(".started"):
                started[record["call_id"]] = record["ts"]
            for key in [
                k for k, v in record["data"].items() if k.endswith("_ms") and isinstance(v, float)
            ]:
                since = started.get(record["call_id"], record["ts"])
                record["data"][key] = round((record["ts"] - since) * 1000, 3)
        run_id = next(r["run_id"] for r in records if r.get("kind") == "meta")
        name = f"{int(first * 1000):013d}.{run_id}.1.jsonl"
        text = "".join(
            json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records
        )
        (out / name).write_text(text, encoding="utf-8")


def build() -> str:
    """The demo page: every run of the example, inline, offline."""
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        runs = record(work / "recorded")
        fixed = work / "fixed"
        fixed.mkdir()
        normalise(runs, fixed)
        folder = RunFolder(fixed)
        folder.refresh()
        summaries = folder.summaries()
        details = {s["key"]: folder.detail(s["key"], paged=False) for s in summaries}
    data = {"runs": summaries, "details": details, "exported_at": BASE_TS}
    return export_html(data)


def main(argv: list[str]) -> int:
    """Write the demo, or with `--check` exit 1 when the committed one is out of date."""
    text = build()
    if "--check" in argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            sys.stderr.write(
                f"{OUT.relative_to(ROOT)} is out of date: run python scripts/gen_docs_demo.py\n"
            )
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(text.encode("utf-8"))
    sys.stdout.write(f"wrote {OUT.relative_to(ROOT).as_posix()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
