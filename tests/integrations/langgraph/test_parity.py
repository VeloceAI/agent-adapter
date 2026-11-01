"""Flow parity: every case of `tests/flow/parity.py` run through `compile_flow`."""

import itertools
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver

from agent_adapter.flow import FlowResult
from agent_adapter.integrations.langgraph import compile_flow, run_compiled
from agent_adapter.integrations.langgraph._spent import CLAIMS
from tests.flow.parity import (
    CASES,
    ParityCase,
    assert_final_matches,
    assert_pauses_match,
    assert_state_matches,
    case,
    run_local,
)
from tests.flow.parity_flows import executor, store

pytestmark = pytest.mark.contract

_threads = itertools.count()


async def run_langgraph(
    parity: ParityCase, *, forget: bool = False, thread: str = ""
) -> tuple[list[FlowResult[Any]], FlowResult[Any]]:
    """Run a case through LangGraph: `(paused results in order, final result)`.

    `forget` resumes each time from a newly compiled graph on the same checkpointer, with the
    process-wide claims cleared (a resume in another process with the same key).
    """
    saver = InMemorySaver()
    app = compile_flow(parity.flow, executor=executor).compile(checkpointer=saver)
    thread = thread or f"parity-{parity.name}-{next(_threads)}"
    config = {"configurable": {"thread_id": thread}, "recursion_limit": parity.max_steps}
    result = await run_compiled(app, parity.initial, config)
    pauses: list[FlowResult[Any]] = []
    for resume in (parity.resume, *parity.then) if parity.resume else ():
        assert result.status == "paused" and result.pending is not None, result.status
        pauses.append(result)
        if forget:
            CLAIMS.clear()
            app = compile_flow(parity.flow, executor=executor).compile(checkpointer=saver)
        result = await run_compiled(app, resume.decision, config)
    return pauses, result


@pytest.mark.parametrize("parity", CASES, ids=[c.name for c in CASES])
async def test_langgraph_matches_each_parity_case(parity):
    if parity.raises is not None:
        with pytest.raises(parity.raises):
            await run_langgraph(parity)
        return
    thread = f"parity-{parity.name}-{next(_threads)}"
    pauses, final = await run_langgraph(parity, thread=thread)
    assert_pauses_match(parity, pauses, durable=False)
    assert not await store.list(run_id=thread, status="pending")  # every record settled
    assert_final_matches(parity, final)


@pytest.mark.parametrize("parity", CASES, ids=[c.name for c in CASES])
async def test_langgraph_and_the_local_runner_end_alike(parity):
    if parity.raises is not None:
        return
    _, local = await run_local(parity)
    _, compiled = await run_langgraph(parity)
    assert (compiled.status, compiled.denied_at) == (local.status, local.denied_at)
    assert dict(compiled.visits) == dict(local.visits)
    assert_state_matches(parity.flow, compiled.state, local.state.model_dump(mode="json"))
    assert compiled.steps == local.steps


async def test_a_resume_in_a_new_process_replays_finished_calls_like_the_local_runner():
    parity = case("replay_two_confirms")
    _, final = await run_langgraph(parity, forget=True)
    assert final.status == "done"
    assert final.state.note == "first#1 second#1"  # the sealed journal replayed "first"


@pytest.mark.parametrize("name", [c.name for c in CASES if c.resume is not None])
async def test_every_pausing_case_resumes_alike_in_a_new_process(name):
    parity = case(name)
    pauses, final = await run_langgraph(parity, forget=True)
    assert_pauses_match(parity, pauses, durable=False)
    assert_final_matches(parity, final)
