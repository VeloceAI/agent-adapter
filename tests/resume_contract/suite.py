"""Every property on every subject, and the results table the docs and the script print."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from tests.resume_contract import flow_scenarios, loop_scenarios
from tests.resume_contract.flow_subject import FlowSubject
from tests.resume_contract.harness import Subject
from tests.resume_contract.loop_subject import LoopSubject
from tests.resume_contract.properties import (
    EXTRA,
    PROPERTIES,
    Matrix,
    Verdict,
    crash_matrix,
    exactly_once,
    lost_record,
    prefix_continuation,
    recovery_determinism,
)

SUBJECTS = ("FlowRunner", "run_tool_loop")


async def matrix_verdicts(make: Callable[[], Subject], m: Matrix) -> list[Verdict]:
    subject = make()
    return [
        prefix_continuation(subject.name, m),
        exactly_once(subject.name, subject.effects, m),
        await recovery_determinism(make, m),
        await lost_record(make, m),
    ]


async def flow_verdicts(tmp: Path, depth: int = 2) -> list[Verdict]:
    m = await crash_matrix(FlowSubject, depth)
    return [
        *await matrix_verdicts(FlowSubject, m),
        await flow_scenarios.fork_determinism(),
        await flow_scenarios.checkpoint_validity(),
        await flow_scenarios.consume_once(tmp),
    ]


async def loop_verdicts(depth: int = 2) -> list[Verdict]:
    m = await crash_matrix(LoopSubject, depth)
    return [
        *await matrix_verdicts(LoopSubject, m),
        await loop_scenarios.fork_determinism(),
        await loop_scenarios.checkpoint_validity(),
        await loop_scenarios.consume_once(),
    ]


def table(verdicts: Sequence[Verdict], subjects: Sequence[str] = SUBJECTS) -> str:
    """A Markdown table: one row per property, one column per subject."""
    cell = {(v.prop, v.subject): v for v in verdicts}
    lines = [
        "| Property | " + " | ".join(subjects) + " |",
        "|---|" + "---|" * len(subjects),
    ]
    for prop in (*PROPERTIES, EXTRA):
        row = []
        for subject in subjects:
            v = cell.get((prop, subject))
            if v is None:
                row.append("not run")
            else:
                mark = "holds" if v.held else f"**fails** ({len(v.failures)})"
                row.append(f"{mark}, {v.cases} cases")
        lines.append(f"| {prop} | " + " | ".join(row) + " |")
    return "\n".join(lines)


async def langgraph_verdicts(depth: int = 1) -> list[Verdict]:
    """The same properties on LangGraph (needs `langgraph`): a comparison, not a gate."""
    from tests.resume_contract import langgraph_scenarios
    from tests.resume_contract.langgraph_subject import LangGraphSubject

    m = await crash_matrix(LangGraphSubject, depth)
    return [
        *await matrix_verdicts(LangGraphSubject, m),
        langgraph_scenarios.fork_determinism(),
        langgraph_scenarios.checkpoint_validity(),
        langgraph_scenarios.consume_once(),
    ]
