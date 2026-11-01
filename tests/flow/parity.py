"""Canonical parity cases: the local runner, LangGraph (WP15) and ADK (WP16) must agree on them.

Each `ParityCase` holds a flow (from `tests.flow.parity_flows`), an initial state and what
every target must end with: status (or the error raised), final state (JSON mode) and
visits per node (completed node runs). Pause cases also give the pause every target must
reach (gate or tool approval, payload, state while paused) and the decision to resume with.
Run every case with `parity_flows.executor`. Compare states with `assert_state_matches`:
`append` fields as multisets, `keyed` fields by key set and values, others by equality.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from agent_adapter.core.errors import FlowLoopExceeded
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.flow import Flow, FlowResult, FlowRunner
from agent_adapter.flow import merge_rules as _merge_rules
from tests.flow.parity_flows import (
    code_tool_flow,
    collision_flow,
    diamond_flow,
    equal_flow,
    executor,
    fanout_flow,
    gate_flow,
    loop_flow,
    map_skip_flow,
    replay_flow,
    review_flow,
    route_view_flow,
    tool_pause_flow,
    two_approvals_flow,
    untaken_join_flow,
)

APPROVE = ApprovalDecision(approved=True)


@dataclass(frozen=True)
class Resume:
    """A pause every target must reach, and the decision that resumes it."""

    node: str  # the paused node
    decision: ApprovalDecision
    state: dict[str, Any]  # the state while paused (the paused superstep's input)
    kind: Literal["node", "tool"] = "node"  # a pause gate, or a tool approval inside the node
    tool: str = ""  # the pending tool (kind "tool")
    payload: dict[str, Any] | None = None  # the pending request's arguments


@dataclass(frozen=True)
class ParityCase:
    """One flow run and the outcome every target must reproduce."""

    name: str
    flow: Flow[Any]
    initial: dict[str, Any]
    status: Literal["done", "denied", "error"]
    expected: dict[str, Any]  # final state, JSON mode ({} when it raises)
    visits: dict[str, int]  # completed runs per node
    resume: Resume | None = None
    then: tuple[Resume, ...] = ()  # further pauses, each reached after resuming the previous
    denied_at: str | None = None
    max_steps: int = 100
    raises: type[Exception] | None = None


def _board(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"flag": "", "note": "", "sent": "", "items": [], "hits": [], "log": []}
    return {**base, **fields}


_PAUSED = {"topic": "flows", "draft": "draft about flows", "published": "", "log": ["write"]}


def _review(decision: ApprovalDecision) -> Resume:
    return Resume("publish", decision, dict(_PAUSED), payload={"draft": "draft about flows"})


_ITEMS = [{"tool": "search", "query": "a"}, {"tool": "broken", "query": "b"}]
_HIT_A = {"source_id": "src-a", "text": "evidence for a"}
_HIT_B = {"source_id": "src-b", "text": "evidence for b"}
_SHARED = {"source_id": "shared", "text": "shared evidence"}
_START = _board(log=["start"])
_SENDS = [{"text": "a"}, {"text": "b"}]


def _stamp_pause(text: str) -> Resume:
    return Resume("go", APPROVE, _board(), kind="tool", tool="stamp", payload={"text": text})


def _send_pause(text: str) -> Resume:
    return Resume(
        "fan", APPROVE, _board(items=_SENDS), kind="tool", tool="send", payload={"text": text}
    )


CASES: tuple[ParityCase, ...] = (
    ParityCase(
        name="fanout_keyed",
        flow=fanout_flow,
        initial={"queries": ["a", "b", "a"]},
        status="done",
        expected={
            "queries": ["a", "b", "a"],
            "plan": [{"tool": "search", "args": {"query": q}} for q in ("a", "b", "a")],
            "evidence": [_HIT_A, _SHARED, _HIT_B],
            "log": ["plan", "count", "audit", "merge"],
            "calls": 4,
            "summary": "shared,src-a,src-b",
        },
        visits={"plan": 1, "retrieve": 1, "count": 1, "audit": 1, "merge": 1},
    ),
    ParityCase(
        name="loop_exhausted",
        flow=loop_flow,
        initial={"target": 10},
        status="done",
        expected={"target": 10, "n": 4, "trail": ["n=1", "n=2", "n=3", "n=4"], "finished": True},
        visits={"work": 4, "note": 4, "finish": 1},
    ),
    ParityCase(
        name="loop_natural",
        flow=loop_flow,
        initial={"target": 2},
        status="done",
        expected={"target": 2, "n": 2, "trail": ["n=1", "n=2"], "finished": True},
        visits={"work": 2, "note": 2, "finish": 1},
    ),
    ParityCase(
        name="max_steps",
        flow=loop_flow,
        initial={"target": 10},
        status="error",
        expected={},
        visits={},
        max_steps=3,
        raises=FlowLoopExceeded,
    ),
    ParityCase(
        name="pause_approved",
        flow=review_flow,
        initial={"topic": "flows"},
        status="done",
        expected={**_PAUSED, "published": "draft about flows", "log": ["write", "publish"]},
        visits={"write": 1, "publish": 1},
        resume=_review(APPROVE),
    ),
    ParityCase(
        name="pause_edited",
        flow=review_flow,
        initial={"topic": "flows"},
        status="done",
        expected={
            **_PAUSED,
            "draft": "edited draft",
            "published": "edited draft",
            "log": ["write", "publish"],
        },
        visits={"write": 1, "publish": 1},
        resume=_review(ApprovalDecision(approved=True, edited_args={"draft": "edited draft"})),
    ),
    ParityCase(
        name="pause_denied",
        flow=review_flow,
        initial={"topic": "flows"},
        status="denied",
        expected=dict(_PAUSED),
        visits={"write": 1},
        resume=_review(ApprovalDecision(approved=False, reason="not yet")),
        denied_at="publish",
    ),
    ParityCase(
        name="gate_sibling_approved",
        flow=gate_flow,
        initial={},
        status="done",
        expected=_board(log=["start", "gated", "side"]),
        visits={"start": 1, "gated": 1, "side": 1},
        resume=Resume("gated", APPROVE, _START, payload={}),
    ),
    ParityCase(
        name="gate_sibling_denied",
        flow=gate_flow,
        initial={},
        status="denied",
        expected=_START,
        visits={"start": 1},  # the sibling did not run either
        resume=Resume("gated", ApprovalDecision(approved=False), _START, payload={}),
        denied_at="gated",
    ),
    ParityCase(
        name="tool_pause_sibling",
        flow=tool_pause_flow,
        initial={},
        status="done",
        expected=_board(sent="sent hello", log=["start", "side"]),  # "side" ran once
        visits={"start": 1, "ask": 1, "side": 1},
        resume=Resume("ask", APPROVE, _START, kind="tool", tool="send", payload={"text": "hello"}),
    ),
    ParityCase(
        name="code_calls_confirm_tool",
        flow=code_tool_flow,
        initial={},
        status="done",
        expected=_board(sent="sent from code", log=["code"]),
        visits={"go": 1},
        resume=Resume(
            "go", APPROVE, _board(), kind="tool", tool="send", payload={"text": "from code"}
        ),
    ),
    ParityCase(
        name="route_sees_own_writes",
        flow=route_view_flow,
        initial={},
        status="done",
        expected=_board(flag="set", log=["start", "setter", "chooser", "no_flag"]),
        visits={"start": 1, "setter": 1, "chooser": 1, "no_flag": 1},
    ),
    ParityCase(
        name="keyed_collision",
        flow=collision_flow,
        initial={},
        status="done",
        expected=_board(hits=[{"source_id": "x", "text": "zeta"}], log=["start", "alpha", "zeta"]),
        visits={"start": 1, "zeta": 1, "alpha": 1},
    ),
    ParityCase(
        name="equal_replace",
        flow=equal_flow,
        initial={},
        status="done",
        expected=_board(note="same", log=["start", "left", "right"]),
        visits={"start": 1, "left": 1, "right": 1},
    ),
    ParityCase(
        name="diamond",
        flow=diamond_flow,
        initial={},
        status="done",
        expected=_board(log=["top", "left", "right", "bottom"]),
        visits={"top": 1, "left": 1, "right": 1, "bottom": 1},
    ),
    ParityCase(
        name="untaken_join",
        flow=untaken_join_flow,
        initial={},
        status="done",
        expected=_board(log=["pick", "taken", "joined"]),
        visits={"pick": 1, "taken": 1, "joined": 1},
    ),
    ParityCase(
        name="two_approvals_one_node",
        flow=two_approvals_flow,
        initial={"items": _SENDS},
        status="done",
        expected=_board(items=_SENDS, log=["sent a", "sent b"]),
        visits={"fan": 1},
        resume=_send_pause("a"),
        then=(_send_pause("b"),),
    ),
    ParityCase(
        name="replay_two_confirms",  # LangGraph re-runs the node: "first#2" there (see docs)
        flow=replay_flow,
        initial={},
        status="done",
        expected=_board(note="first#1 second#1"),
        visits={"go": 1},
        resume=_stamp_pause("first"),
        then=(_stamp_pause("second"),),
    ),
    ParityCase(
        name="map_skip",
        flow=map_skip_flow,
        initial={"items": _ITEMS},
        status="done",
        expected=_board(items=_ITEMS, hits=[_HIT_A, _SHARED]),
        visits={"fan": 1},
    ),
)


def case(name: str) -> ParityCase:
    """The case called `name`."""
    return next(c for c in CASES if c.name == name)


def assert_state_matches(
    flow: Flow[Any], actual: BaseModel | Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Assert `actual` equals `expected` under the parity comparison of each field's merge rule."""
    data = actual.model_dump(mode="json") if isinstance(actual, BaseModel) else dict(actual)
    rules = _merge_rules(flow.state_model)
    assert set(data) == set(expected), f"fields differ: {sorted(data)} != {sorted(expected)}"
    for name, want in expected.items():
        got, rule = data[name], rules[name]
        by = getattr(rule, "by", None)
        if rule.name == "append":
            same = _multiset(got) == _multiset(want)
        elif by is not None:
            same = {item[by]: item for item in got} == {item[by]: item for item in want}
        else:
            same = got == want
        assert same, f"field {name!r} ({rule.name}): {got!r} != {want!r}"


def _multiset(values: list[Any]) -> Counter[str]:
    return Counter(json.dumps(value, sort_keys=True) for value in values)


async def run_local(parity: ParityCase) -> tuple[list[FlowResult[Any]], FlowResult[Any]]:
    """Run a case on the local runner: `(paused results in order, final result)`."""
    runner = FlowRunner(executor, max_steps=parity.max_steps)
    result = await runner.run(parity.flow, parity.initial)
    pauses: list[FlowResult[Any]] = []
    for resume in (parity.resume, *parity.then) if parity.resume else ():
        assert result.status == "paused" and result.checkpoint is not None, result.status
        pauses.append(result)
        result = await runner.resume(parity.flow, result.checkpoint, resume.decision)
    return pauses, result


def assert_pauses_match(
    parity: ParityCase, pauses: list[FlowResult[Any]], *, durable: bool = True
) -> None:
    """Assert each pause asks what the case expects; `durable=False` wants no checkpoint."""
    expected = (parity.resume, *parity.then) if parity.resume else ()
    assert len(pauses) == len(expected)
    for paused, resume in zip(pauses, expected, strict=True):
        pending = paused.pending
        assert pending is not None
        assert pending.kind == resume.kind and pending.metadata["node"] == resume.node
        if resume.kind == "node":
            assert pending.tool == f"node:{parity.flow.name}.{resume.node}"
        else:
            assert pending.tool == resume.tool
        if resume.payload is not None:  # the approver sees the real request
            assert pending.arguments == resume.payload
        assert_state_matches(parity.flow, paused.state, resume.state)
        if not durable:  # durability belongs to the host framework
            assert paused.checkpoint is None
    assert len({p.pending.approval_id for p in pauses if p.pending}) == len(pauses)


def assert_final_matches(parity: ParityCase, final: FlowResult[Any]) -> None:
    """Assert the final result has the case's status, denial, visits and state."""
    assert final.status == parity.status
    assert final.denied_at == parity.denied_at
    assert dict(final.visits) == parity.visits
    assert_state_matches(parity.flow, final.state, parity.expected)
