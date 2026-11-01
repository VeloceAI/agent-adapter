"""Optimise the help-center FAQ agent offline, then load and apply the tree it chose."""

import asyncio
import tempfile
from pathlib import Path

from harness_agent import SURFACE, run_case
from harness_fakes import FREE, FakeFaqModel, ScriptedProposer

from agent_adapter import Ledger
from agent_adapter.optimiser import (
    CallableHost,
    OptimiserBudget,
    OptimiserConfig,
    load_cases,
    load_harness,
    optimise,
)
from agent_adapter.optimiser.graders import contains

CASES = Path(__file__).with_name("harness_cases.jsonl")  # swap in your own cases file


def build_host() -> CallableHost:
    """The host for the CLI: `--host harness_optimise:build_host`."""
    # --8<-- [start:grader]
    judge = contains()  # 1.0 when the expected article id is in the answer
    # --8<-- [end:grader]
    # --8<-- [start:host]
    host = CallableHost(
        run_case,  # your agent: (tree, case, model) -> answer text
        surface=SURFACE,  # the texts it may tune
        cases=load_cases(CASES),
        judge=judge,
        domain="support",
        required=["prompt/system"],
    )
    # --8<-- [end:host]
    return host


async def main() -> None:
    host = build_host()
    with tempfile.TemporaryDirectory() as out:
        # --8<-- [start:run]
        config = OptimiserConfig(
            k=1, out_dir=out, lessons_path="", budget=OptimiserBudget(max_case_runs=40)
        )
        result = await optimise(
            host,
            config,
            proposer=ScriptedProposer(),  # swap in your proposer model here
            model=FakeFaqModel(),  # swap in your agent's model here
            ledger=Ledger(prices=FREE),  # your ledger and price table; optional
        )
        # --8<-- [end:run]
        final = result.chosen or "base"
        after = result.probes[final].mean_score if result.chosen else result.base.mean_score
        print(f"base {result.base.mean_score:.2f} -> {final} {after:.2f}  (chosen: {final})")
        for line in result.decisions:
            print(f"{line.kind:<7} {line.path}")
        print(result.patch.splitlines()[0])
        # --8<-- [start:apply]
        tree = load_harness(out)  # the tree you reviewed in harness.patch
        print(tree.files["prompt/system"])  # feed it to your agent, as run_case does
        # --8<-- [end:apply]


if __name__ == "__main__":
    asyncio.run(main())
