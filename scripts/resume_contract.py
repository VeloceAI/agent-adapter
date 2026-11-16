"""resume-contract: the six resume properties of arXiv 2608.03836, under injected crashes.

Deterministic and offline. Crashes every durable run at every step boundary (each store write,
before and after, and inside each tool and model call), and with `--depth 2` at every pair of
them, then checks prefix continuation, effect exactly-once, fork determinism, checkpoint
validity, consume-once and recovery determinism on `FlowRunner` and `run_tool_loop`.

    python scripts/resume_contract.py            # ours, depth 2
    python scripts/resume_contract.py --langgraph  # plus LangGraph, if installed (depth 1)

Exits 1 when a property fails for agent-adapter.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.resume_contract.properties import Verdict
from tests.resume_contract.suite import (
    SUBJECTS,
    flow_verdicts,
    langgraph_verdicts,
    loop_verdicts,
    table,
)


async def _run(depth: int, langgraph: bool) -> tuple[list[Verdict], list[Verdict]]:
    with tempfile.TemporaryDirectory() as tmp:
        ours = await flow_verdicts(Path(tmp), depth) + await loop_verdicts(depth)
    theirs = await langgraph_verdicts() if langgraph else []
    return ours, theirs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--depth", type=int, default=2, choices=(1, 2), help="crashes per run")
    parser.add_argument("--langgraph", action="store_true", help="add LangGraph (depth 1)")
    args = parser.parse_args()
    logging.disable(logging.ERROR)  # recoveries log every refused store and call in doubt
    started = time.perf_counter()
    if args.langgraph:
        try:
            import langgraph  # noqa: F401
        except ImportError:
            print("LangGraph is not installed: pip install langgraph", file=sys.stderr)
            return 2
    ours, theirs = asyncio.run(_run(args.depth, args.langgraph))
    subjects = [*SUBJECTS, "LangGraph"] if theirs else list(SUBJECTS)
    print(table([*ours, *theirs], subjects))
    print()
    for v in [*ours, *theirs]:
        print(f"- {v.subject}, {v.prop}: {v.note}")
        for failure in v.failures[:3]:
            print(f"    {failure}")
    print(f"\ndepth {args.depth}, {time.perf_counter() - started:.0f} s")
    return 0 if all(v.held for v in ours) else 1


if __name__ == "__main__":
    sys.exit(main())
