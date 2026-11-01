"""The optimiser loop when inputs, the proposer, the agent or the clock fail."""

import asyncio
import json
import time
from typing import ClassVar

import pytest

from agent_adapter.cache.memory import MemoryCache
from agent_adapter.core.errors import (
    ConfigError,
    ModelAuthError,
    ModelBadRequest,
    ModelUnavailable,
    OptimiserError,
)
from agent_adapter.optimiser import OptimiserBudget
from agent_adapter.optimiser.lessons import LessonsStore
from tests.optimiser.fakes import (
    FakeHost,
    ToyAgent,
    keep_base,
    proposal,
    system_of,
    user_of,
)
from tests.optimiser.loop_support import CITE, HINT, LESSON, run


async def test_bad_inputs_fail_before_any_call(tmp_path):
    with pytest.raises(ConfigError, match="k must be >= 1"):
        await run(tmp_path, {}, k=0)
    with pytest.raises(ConfigError, match=r"budget\.max_usd must be >= 0"):
        await run(tmp_path, {}, budget=OptimiserBudget(max_usd=-1))
    with pytest.raises(OptimiserError, match="no eval cases"):
        await run(tmp_path, {}, host=FakeHost(cases=()))
    with pytest.raises(OptimiserError, match="surface is invalid"):
        await run(tmp_path, {}, host=FakeHost(files={"prompt/system": "x" * 13_000}))
    (tmp_path / "lessons.json").write_text("{broken", "utf-8")
    with pytest.raises(OptimiserError, match="lessons file"):
        await run(tmp_path, {})


async def test_a_wall_clock_limit_cancels_a_probe_in_flight(tmp_path):
    class Hangs(FakeHost):
        async def run(self, tree, case, *, run_id, model):
            await asyncio.sleep(10)

    started = time.monotonic()
    result, *_ = await run(tmp_path, {}, host=Hangs(), budget=OptimiserBudget(max_wall_s=0.05))
    assert result.budget.stopped == "max_wall_s"
    assert time.monotonic() - started < 5


async def test_case_timeout_reaches_the_probe(tmp_path):
    class Hangs(FakeHost):
        async def run(self, tree, case, *, run_id, model):
            await asyncio.sleep(10)

    with pytest.raises(OptimiserError, match="no base case run succeeded"):
        await run(tmp_path, {}, host=Hangs(), case_timeout_s=0.01)


@pytest.mark.parametrize(
    "error", [ModelAuthError("bad key", provider="openai"), ModelBadRequest("x", provider="openai")]
)
async def test_fatal_proposer_errors_raise(tmp_path, error):
    with pytest.raises(OptimiserError, match=type(error).kind):
        await run(tmp_path, {"proc1": error, "know1": keep_base()})


async def test_a_bad_agent_key_raises(tmp_path):
    from agent_adapter.models.callable import CallableModel

    async def denied(request):
        raise ModelAuthError("invalid key", provider="openai")

    class Agent:
        port = CallableModel(denied)
        requests: ClassVar[list] = []

    with pytest.raises(OptimiserError, match="model-auth-failed"):
        await run(tmp_path, {}, agent=Agent())


async def test_every_proposer_failing_on_a_model_error_raises(tmp_path):
    down = ModelUnavailable("down", provider="openai")
    with pytest.raises(OptimiserError, match="every proposer failed on a model error"):
        await run(tmp_path, {"proc1": down, "know1": down})


async def test_a_crash_after_reflection_does_not_duplicate_the_lesson(tmp_path, monkeypatch):
    real_add = LessonsStore.add
    calls = []

    def flaky(self, lessons):
        calls.append(len(lessons))
        if len(calls) == 1:
            raise OSError("disk full")
        return real_add(self, lessons)

    monkeypatch.setattr(LessonsStore, "add", flaky)
    script = {"proc1": keep_base(), "know1": proposal(CITE), "reflect": LESSON}
    with pytest.raises(OSError):
        await run(tmp_path, script)
    other = {**LESSON, "cue": "a different cue"}
    _, proposer, *_ = await run(
        tmp_path, {"proc1": keep_base(), "know1": proposal(CITE), "reflect": other}
    )
    assert "reflect" not in proposer.stages()
    stored = json.loads((tmp_path / "lessons.json").read_text("utf-8"))["lessons"]
    assert [x["cue"] for x in stored] == [LESSON["cue"]]


def bad_request(request=None):
    return ModelBadRequest("rejected", provider="openai")


async def test_a_transient_proposer_failure_resumes_and_proposes_again(tmp_path):
    down = ModelUnavailable("down", provider="openai")
    cache = MemoryCache()
    with pytest.raises(OptimiserError, match="every proposer failed on a model error"):
        await run(tmp_path, {"proc1": down, "know1": down}, cache=cache)
    result, proposer, *_ = await run(
        tmp_path, {"proc1": keep_base(), "know1": proposal(CITE)}, cache=cache
    )
    assert proposer.stages()[:2] == ["propose:proc1", "propose:know1"]
    assert result.chosen == "agg1"
    assert (result.budget.proposer_calls, result.budget.case_runs) == (2, 12)


async def test_an_agent_rejecting_every_candidate_request_fails_the_run_visibly(tmp_path):
    cache = MemoryCache()
    await run(
        tmp_path,
        {"proc1": keep_base(), "know1": keep_base()},
        cache=cache,
        out_dir=str(tmp_path / "warm"),
        lessons_path="",
    )
    rejecting = ToyAgent(reject=bad_request)  # the base is still served from the cache
    with pytest.raises(OptimiserError, match="every candidate case run failed"):
        await run(
            tmp_path,
            {"proc1": proposal(HINT), "know1": proposal(CITE)},
            cache=cache,
            agent=rejecting,
            out_dir=str(tmp_path / "cold"),
            lessons_path="",
            seed=1,  # new proposer requests; the agent's base requests still hit the cache
        )
    report = (tmp_path / "cold" / "report.md").read_text("utf-8")
    assert "proc1: 3/3 case runs failed: model-bad-request" in report
    assert "know1: 3/3 case runs failed: model-bad-request" in report


async def test_failed_case_runs_of_a_candidate_are_listed_as_errors(tmp_path):
    def reject(request):
        return bad_request() if "REJECT" in system_of(request) else None

    result, *_ = await run(
        tmp_path,
        {"proc1": keep_base(), "know1": proposal(("prompt/system", "REJECT but cite"))},
        agent=ToyAgent(reject=reject),
    )
    assert "know1: 3/3 case runs failed: model-bad-request" in result.errors
    assert "know1: 3/3 case runs failed: model-bad-request" in result.report_md


async def test_a_bad_request_in_some_base_cases_is_recorded_not_fatal(tmp_path):
    def reject(request):
        return bad_request() if user_of(request) == "tuition fees" else None

    result, *_ = await run(
        tmp_path, {"proc1": keep_base(), "know1": proposal(CITE)}, agent=ToyAgent(reject=reject)
    )
    assert not result.base.runs[1].ok
    assert "base: 1/3 case runs failed: model-bad-request" in result.errors


async def test_a_bad_request_in_every_base_case_is_fatal_and_names_the_case(tmp_path):
    with pytest.raises(OptimiserError, match=r"probe base, case q1: model-bad-request"):
        await run(tmp_path, {}, agent=ToyAgent(reject=bad_request))


async def test_an_auth_error_in_one_base_case_is_fatal_and_names_the_case(tmp_path):
    def reject(request):
        if user_of(request) == "home loan":
            return ModelAuthError("expired key", provider="openai")
        return None

    with pytest.raises(OptimiserError, match=r"probe base, case q3: model-auth-failed"):
        await run(tmp_path, {}, agent=ToyAgent(reject=reject))


async def test_the_report_counts_restored_reflection_and_selection_calls(tmp_path):
    script = {
        "proc1": keep_base(),
        "know1": proposal(CITE),
        "select": {"why": "w", "choice": "A"},
        "reflect": LESSON,
    }
    await run(tmp_path, script, host=FakeHost(judge=None))
    again, proposer, *_ = await run(tmp_path, script, host=FakeHost(judge=None))
    assert proposer.requests == []
    assert "| reflection calls | 1 | at most 1 |" in again.report_md
    assert "| selection calls | 1 | at most 1, only without a judge |" in again.report_md


async def test_candidates_that_time_out_are_listed_and_end_as_keep_base(tmp_path):
    host = FakeHost()
    base_tree = host.surface()

    class SlowCandidates(FakeHost):
        async def run(self, tree, case, *, run_id, model):
            if tree != base_tree:
                await asyncio.sleep(10)  # cancelled by case_timeout_s
            return await super().run(tree, case, run_id=run_id, model=model)

    result, *_ = await run(
        tmp_path,
        {"proc1": proposal(HINT), "know1": proposal(CITE)},
        host=SlowCandidates(),
        case_timeout_s=0.05,
    )
    assert result.chosen is None  # a slow but healthy host is not a fatal error
    assert "proc1: 3/3 case runs failed: case-timeout" in result.errors
    assert "know1: 3/3 case runs failed: case-timeout" in result.errors
    report = (tmp_path / "out" / "report.md").read_text("utf-8")
    assert "proc1: 3/3 case runs failed: case-timeout" in report
    assert "raise `case_timeout_s`" in report
