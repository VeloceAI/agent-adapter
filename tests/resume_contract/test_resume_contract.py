"""The six resume properties of arXiv 2608.03836 must hold for `FlowRunner` and `run_tool_loop`.

Every crash point is tried once (depth 1); `AA_RESUME_DEPTH=2` adds every pair of crashes,
as `scripts/resume_contract.py` does by default.
"""

from __future__ import annotations

import asyncio
import functools
import os
from pathlib import Path

import pytest

from tests.resume_contract.properties import EXTRA, PROPERTIES, Verdict
from tests.resume_contract.suite import flow_verdicts, loop_verdicts, table

DEPTH = int(os.environ.get("AA_RESUME_DEPTH", "1"))


@functools.cache
def _verdicts(subject: str, tmp: str) -> dict[str, Verdict]:
    run = flow_verdicts(Path(tmp), DEPTH) if subject == "FlowRunner" else loop_verdicts(DEPTH)
    return {v.prop: v for v in asyncio.run(run)}


@pytest.fixture(scope="module")
def tmp(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp("resume_contract"))


@pytest.mark.parametrize("prop", [*PROPERTIES, EXTRA])
@pytest.mark.parametrize("subject", ["FlowRunner", "run_tool_loop"])
def test_the_property_holds(subject: str, prop: str, tmp: str) -> None:
    verdict = _verdicts(subject, tmp)[prop]
    assert verdict.cases > 0
    assert verdict.failures == [], "\n".join(verdict.failures[:10])


def test_the_table_has_a_row_per_property(tmp: str) -> None:
    verdicts = [*_verdicts("FlowRunner", tmp).values(), *_verdicts("run_tool_loop", tmp).values()]
    rows = table(verdicts).splitlines()
    assert len(rows) == 2 + len(PROPERTIES) + 1
    assert all("holds" in row for row in rows[2:])
