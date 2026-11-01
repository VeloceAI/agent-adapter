"""Semi-public executor seam for framework integrations (LangGraph, ADK); not in `exec.__all__`.

Integrations that drive the executor's approval, journal and resume internals import them
from here, never from the private modules, so a refactor there keeps this surface stable.
"""

from agent_adapter.exec._approve import _settle_own, approval_store, same_decision
from agent_adapter.exec._canon import NotSealable, check_plain, decode, encode
from agent_adapter.exec._journal import (
    RAN_UNKEYED,
    Journal,
    JournalData,
    _make_journal,
    unsealable_calls,
)
from agent_adapter.exec._resume import AnswerEntry, Pending, _make_answer, _visit_scope
from agent_adapter.exec.parallel import _gather_unwrapped

settle_own = _settle_own
"""Decide and consume a resumed answer's record in the executor's own store, if it is there."""

make_answer = _make_answer
"""The `ResumeAnswer` for a visit from its `AnswerEntry`s."""

visit_scope = _visit_scope
"""Context manager installing a resume answer and a journal for one node visit."""

make_journal = _make_journal
"""A `Journal` for one node visit (replays finished calls when the node re-runs)."""

gather_unwrapped = _gather_unwrapped
"""Run factories concurrently in input order; the first escaping exception cancels the rest."""

__all__ = [
    "RAN_UNKEYED",
    "AnswerEntry",
    "Journal",
    "JournalData",
    "NotSealable",
    "Pending",
    "approval_store",
    "check_plain",
    "decode",
    "encode",
    "gather_unwrapped",
    "make_answer",
    "make_journal",
    "same_decision",
    "settle_own",
    "unsealable_calls",
    "visit_scope",
]
