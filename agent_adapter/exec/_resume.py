"""What a flow node visit hands the executor: resumed approvals and the replay journal.

Only the flow runner makes these objects (`_make_answer`, `exec._journal._make_journal`) and
installs them for one node visit (`_visit_scope`); the executor honours exactly these types.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from agent_adapter.approval.store import ApprovalStore, request_digest
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec._journal import Journal

logger = logging.getLogger(__name__)

_TOKEN = object()  # module-private: only `_make_answer` constructs a ResumeAnswer

AnswerEntry = tuple[ApprovalRequest, ApprovalDecision, bool]  # request, decision, exact call id
Pending = tuple[ApprovalRequest, ApprovalStore | None]  # a pending request and its port's store


class ResumeAnswer:
    """Decisions a flow resume hands back to the requests they were given for, each once.

    An entry answers a request only for the same run, tool and original arguments
    (`request_digest`), and, when `exact`, the same call id. A closed answer answers nothing.
    """

    def __init__(self, token: object, entries: Iterable[AnswerEntry]) -> None:
        if token is not _TOKEN:
            raise ToolUsageError("ResumeAnswer is made by the flow runner only")
        self._given: list[_Answer | None] = []  # one per entry given; None: cannot answer
        for request, decision, exact in entries:
            try:
                digest = request_digest(request)
            except ToolUsageError:
                logger.warning("resumed approval %s has non-JSON arguments", request.approval_id)
                self._given.append(None)
                continue
            self._given.append(_Answer(request, decision, exact, digest))
        self._entries = [entry for entry in self._given if entry is not None]
        self._lock = threading.Lock()
        self._closed = False

    def __call__(self, request: ApprovalRequest) -> tuple[ApprovalRequest, ApprovalDecision] | None:
        """The (original request, decision) answering `request`, used up; else None."""
        try:
            digest = request_digest(request)
        except ToolUsageError:
            return None
        with self._lock:
            if self._closed:
                return None
            for entry in self._entries:
                if entry.matches(request, digest):
                    entry.used = True
                    logger.debug(
                        "approval %s resumed for call %s",
                        entry.request.approval_id,
                        request.call_id,
                    )
                    return entry.request, entry.decision
        return None

    def used(self) -> list[bool]:
        """For each entry given, in order: whether it answered a request (the call went on)."""
        with self._lock:
            return [entry is not None and entry.used for entry in self._given]

    def close(self) -> None:
        """Answer nothing from now on (tasks that outlive the resumed node ask the port)."""
        with self._lock:
            self._closed = True


@dataclasses.dataclass
class _Answer:
    request: ApprovalRequest
    decision: ApprovalDecision
    exact: bool
    digest: str
    used: bool = False

    def matches(self, request: ApprovalRequest, digest: str) -> bool:
        pending = self.request
        return (
            not self.used
            and request.run_id == pending.run_id
            and request.tool == pending.tool
            and digest == self.digest
            and (not self.exact or request.call_id == pending.call_id)
        )


@dataclasses.dataclass(frozen=True)
class _Visit:
    answer: ResumeAnswer | None
    journal: Journal | None
    pendings: list[Pending] = dataclasses.field(default_factory=list)  # every pending it made


_VISIT: ContextVar[_Visit | None] = ContextVar("aa_flow_visit", default=None)


def _make_answer(entries: Iterable[AnswerEntry]) -> ResumeAnswer:
    return ResumeAnswer(_TOKEN, entries)


@contextmanager
def _visit_scope(answer: ResumeAnswer | None, journal: Journal | None) -> Iterator[list[Pending]]:
    """Install `answer` and `journal` for one node visit (and its child tasks), then close them.

    Yields the list every approval request left pending in the visit is appended to.
    """
    visit = _Visit(answer, journal)
    token = _VISIT.set(visit)
    try:
        yield visit.pendings
    finally:
        _VISIT.reset(token)
        for part in (answer, journal):
            if part is not None:
                part.close()


def _current() -> _Visit | None:
    visit = _VISIT.get()
    return visit if type(visit) is _Visit else None


def resumed_answer(request: ApprovalRequest) -> tuple[ApprovalRequest, ApprovalDecision] | None:
    """The answered request and decision for `request` in this visit, or None (ask the port)."""
    visit = _current()
    if visit is None or type(visit.answer) is not ResumeAnswer:
        return None
    return visit.answer(request)


@contextmanager
def _journal_scope(journal: Journal) -> Iterator[None]:
    """Use `journal` (a child journal) in place of the visit's one, keeping its answer."""
    visit = _current()
    answer, pendings = (visit.answer, visit.pendings) if visit is not None else (None, [])
    token = _VISIT.set(_Visit(answer, journal, pendings))
    try:
        yield
    finally:
        _VISIT.reset(token)
        journal.close()


def record_pending(request: ApprovalRequest, store: ApprovalStore | None) -> None:
    """Note a request the port left pending, so the visit's pause can account for it."""
    visit = _current()
    if visit is not None:
        visit.pendings.append((request, store))


def current_journal() -> Journal | None:
    """The replay journal of the running node visit, or None outside flows."""
    visit = _current()
    if visit is None or type(visit.journal) is not Journal:
        return None
    return visit.journal


def visit_snapshot() -> object:
    """The visit scope now, to put back after a tool body (`restore_visit`)."""
    return _VISIT.get()


def restore_visit(saved: object) -> None:
    """Undo a change a tool body made to the visit scope, so its caller never sees it."""
    if _VISIT.get() is not saved:
        logger.warning("a tool body changed the flow visit scope; restored")
        _VISIT.set(saved if type(saved) is _Visit else None)
