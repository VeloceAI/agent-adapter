"""LangGraph scenarios: checkpoint validity, consume-once and fork determinism."""

from __future__ import annotations

import contextlib
import operator
import threading
from typing import Annotated, Any

from langgraph.types import Command
from pydantic import BaseModel

from tests.resume_contract import langgraph_subject as lg
from tests.resume_contract.harness import ACTIVE, Bank, World
from tests.resume_contract.properties import PROPERTIES, Verdict

NAME = "LangGraph"


def _fresh() -> Bank:
    ACTIVE.world, ACTIVE.bank = World(), Bank()
    return ACTIVE.bank


def _paused() -> tuple[lg.FaultSaver, Any]:
    saver = lg.FaultSaver()
    app = lg.GRAPH.compile(checkpointer=saver)
    app.invoke({"order": "o-1", "log": []}, lg.CONFIG, durability=lg.DURABILITY)
    return saver, app


def _resume(app: Any, value: Any, config: Any = lg.CONFIG) -> Any:
    try:
        result = app.invoke(Command(resume=value), config, durability=lg.DURABILITY)
        return ("paused" if "__interrupt__" in result else "done", result["log"])
    except Exception as exc:  # the outcome is the error type
        return ("error", type(exc).__name__)


def fork_determinism() -> Verdict:
    """Forks by byte copy, and by LangGraph's own fork intent: resume an older checkpoint id."""
    failures: list[str] = []
    values = {"amount 5": 5, "amount 9": 9, "deny": 0}
    alone = {}
    for name, value in values.items():
        bank = _fresh()
        _, app = _paused()
        alone[name] = (_resume(app, value), bank.applied["refund"])
    _fresh()
    saver, _ = _paused()
    for name, value in values.items():
        bank = _fresh()
        app = lg.GRAPH.compile(checkpointer=saver.snapshot())
        got = (_resume(app, value), bank.applied["refund"])
        if got != alone[name]:
            failures.append(f"byte-copy fork {name}: {got}, alone {alone[name]}")
    _fresh()
    saver, app = _paused()
    at_interrupt = saver.get_tuple(lg.CONFIG).config  # the fork point
    _resume(app, 5)
    for name, value in values.items():
        bank = _fresh()
        got = (_resume(app, value, at_interrupt), bank.applied["refund"])
        if got != alone[name]:
            failures.append(
                f"fork from the interrupt checkpoint, {name}: {got}, alone {alone[name]}"
            )
    note = "byte-copy forks hold; a resume of the interrupt's checkpoint id with a new value"
    note += " does not follow that value" if failures else " follows it"
    return Verdict(PROPERTIES[2], NAME, 2 * len(values), failures, note)


class TripModel(BaseModel):
    order: str = "o-1"
    log: Annotated[list[str], operator.add] = []


def first(state: TripModel) -> dict[str, Any]:
    return {"log": ["first"]}


def wrong_type(state: TripModel) -> dict[str, Any]:
    return {"order": 7}


def in_place(state: TripModel) -> dict[str, Any]:
    state.log.append(7)  # type: ignore[arg-type]
    return {}


def _stored(saver: lg.FaultSaver) -> dict[str, Any]:
    found = saver.get_tuple(lg.CONFIG)
    return dict(found.checkpoint["channel_values"]) if found is not None else {}


def _valid(values: dict[str, Any]) -> bool:
    try:
        TripModel.model_validate({k: values[k] for k in ("order", "log") if k in values})
        return True
    except ValueError:
        return False


def checkpoint_validity() -> Verdict:
    """A schema-invalid write (node update, in-place change, `update_state`) is refused."""
    failures: list[str] = []
    for name, node in (("wrong type", wrong_type), ("mutated in place", in_place)):
        _fresh()
        saver = lg.FaultSaver()
        app = lg.build(TripModel, [("ok", first), ("bad", node)]).compile(checkpointer=saver)
        try:
            app.invoke({"order": "o-1"}, lg.CONFIG, durability=lg.DURABILITY)
            raised = False
        except Exception:
            raised = True
        if not _valid(_stored(saver)):
            failures.append(f"{name}: stored {_stored(saver)} (raised: {raised})")
    _fresh()
    saver = lg.FaultSaver()
    app = lg.build(TripModel, [("ok", first)]).compile(checkpointer=saver)
    app.invoke({"order": "o-1"}, lg.CONFIG, durability=lg.DURABILITY)
    with contextlib.suppress(Exception):  # refusing it is the right answer
        app.update_state(lg.CONFIG, {"order": 7})
    if not _valid(_stored(saver)):
        failures.append(f"update_state: stored {_stored(saver)}")
    note = "pydantic state; invalid writes are stored without an error" if failures else ""
    return Verdict(PROPERTIES[3], NAME, 3, failures, note or "invalid writes are refused")


def _rival_resumes() -> list[str]:
    bank = _fresh()
    saver, _ = _paused()
    lg.RACE = threading.Barrier(2)
    results: list[Any] = []

    def resume() -> None:
        results.append(_resume(lg.GRAPH.compile(checkpointer=saver), 5))

    threads = [threading.Thread(target=resume) for _ in range(2)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
    finally:
        lg.RACE = None
    if bank.attempts["refund"] != 1:
        return [f"two rival resumes: the refund ran {bank.attempts['refund']} times ({results})"]
    return []


def _redelivered() -> list[str]:
    bank = _fresh()
    _, app = _paused()
    first_end = _resume(app, 5)
    failures = []
    for label, value in (("the same resume again", 5), ("another answer after it", 9)):
        before = dict(bank.attempts)
        again = _resume(app, value)
        if dict(bank.attempts) != before or (again[0] == "done" and again != first_end):
            failures.append(f"{label}: {again}, effects {dict(bank.attempts)}")
    return failures


def consume_once() -> Verdict:
    """One interrupt is consumed by one resume; a redelivered one has no effect."""
    failures = _rival_resumes() + _redelivered()
    note = "judged by effects: a refused or a silently ignored resume both count as no effect"
    return Verdict(PROPERTIES[4], NAME, 3, failures, note)
