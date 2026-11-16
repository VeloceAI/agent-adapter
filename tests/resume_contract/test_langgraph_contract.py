"""The same resume properties on LangGraph with a checkpointer: a comparison, not a gate.

It asserts what LangGraph does today, so a change in LangGraph shows up here; update
`docs/concepts/resume-contract.md` with it.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("langgraph")

from tests.resume_contract.suite import langgraph_verdicts

pytestmark = pytest.mark.contract

SEEN = {  # LangGraph 1.2.x, InMemorySaver, durability="sync"
    "prefix continuation": True,
    "effect exactly-once": False,
    "fork determinism": False,
    "checkpoint validity": False,
    "consume-once": False,
    "recovery determinism": True,
    "lost record refused (extra)": False,
}


def test_langgraph_shows_what_the_docs_say() -> None:
    verdicts = asyncio.run(langgraph_verdicts())
    assert {v.prop: v.held for v in verdicts} == SEEN
