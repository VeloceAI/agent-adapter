import asyncio
import json
import logging

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import Event
from agent_adapter.evals import (
    CASE_EVENT,
    EvalCase,
    EvalReport,
    EvalRun,
    LLMJudge,
    Outcome,
    Score,
    _collect,
    eval_executor,
    run_eval,
)
from agent_adapter.evals._collect import Collector, case_cost
from agent_adapter.evals.graders import (
    all_of,
    code_check,
    contains,
    exact,
    max_cost,
    max_latency,
    regex,
    tools_called,
)
from agent_adapter.evals.runner import run_eval_sync
from agent_adapter.exec.executor import Executor
from agent_adapter.viewer import record
from tests.evals.fakes import case, shop_registry
from tests.fakes.sinks import RecordingSink


def shipped(question):
    return "shipped"


async def test_each_case_is_a_recorded_run_with_its_own_id():
    sink = RecordingSink()
    executor = Executor(shop_registry(), sink=sink)
    report = await run_eval(
        shipped, [case("a"), case("b", tags=("x",))], [exact()], executor=executor, name="shop"
    )
    ids = [c.run_id for c in report.cases]
    assert len(set(ids)) == 2 and all(i.startswith("run_") for i in ids)
    for run_id in ids:
        types = [e.type for e in sink.events if e.run_id == run_id]
        assert types == ["run.started", CASE_EVENT, "run.finished"]
    marker = next(e for e in sink.events if e.type == CASE_EVENT)
    assert dict(marker.data) == {"eval": "shop", "case_id": "a", "passed": True, "score": 1.0}
    assert report.name == "shop" and report.cases[1].tags == ("x",)


async def test_the_viewer_recorder_records_each_case(tmp_path):
    executor = eval_executor()
    recorder = record(executor, folder=tmp_path)
    report = await run_eval(shipped, [case()], [exact()], executor=executor)
    recorder.stop()
    written = "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.jsonl"))
    assert report.cases[0].run_id in written


async def test_a_failing_case_records_a_safe_error_and_skips_grading(caplog):
    caplog.set_level(logging.WARNING)
    graded = []

    async def grader(c, r):
        graded.append(c.id)
        return Score(value=1.0, passed=True)

    def boom(question):
        raise RuntimeError("secret text")

    def typed(question):
        raise ConfigError("no such shop")

    sink = RecordingSink()
    report = await run_eval(boom, [case()], [grader], executor=Executor(shop_registry(), sink=sink))
    result = report.cases[0]
    assert result.error == "internal-error: RuntimeError" and not result.passed
    assert graded == [] and "secret text" not in caplog.text
    finished = next(e for e in sink.events if e.type == "run.finished")
    assert finished.data["ok"] is False
    typed_error = (await run_eval(typed, [case()], [grader])).cases[0].error
    assert typed_error == "config-invalid: no such shop"


async def test_a_slow_case_times_out():
    async def slow(question):
        await asyncio.sleep(5)

    result = (await run_eval(slow, [case()], [exact()], timeout_s=0.05)).cases[0]
    assert result.error == "timeout: the case ran longer than 0.05s"


async def test_unfinished_outcomes_keep_their_output():
    class Half:
        name = "half"

        async def run_case(self, case, *, executor, ctx):
            return Outcome("partial", unfinished="tool loop max_steps")

    result = (await run_eval(Half(), [case()], [exact()])).cases[0]
    assert result.output == "partial" and result.error == "unfinished: tool loop max_steps"


async def test_grader_errors_and_bad_results_fail_the_case(caplog):
    caplog.set_level(logging.WARNING)

    async def raises(c, r):
        raise ValueError("leak")

    async def odd(c, r):
        return 0.5

    report = await run_eval(shipped, [case()], {"raises": raises, "odd": odd, "ok": exact()})
    grades = {g.grader: g for g in report.cases[0].grades}
    assert grades["raises"].score is None and grades["raises"].notes == "not graded"
    assert grades["odd"].score is None and grades["ok"].passed
    assert not report.cases[0].passed and report.cases[0].score == pytest.approx(1 / 3)
    assert "leak" not in caplog.text and "returned float" in caplog.text


async def test_value_only_scores_pass_at_pass_at():
    async def half(c, r):
        return Score(value=0.6)

    strict = await run_eval(shipped, [case()], [half])
    lenient = await run_eval(shipped, [case()], [half], pass_at=0.5)
    assert not strict.cases[0].passed and lenient.cases[0].passed


async def test_duplicate_grader_names_are_numbered():
    report = await run_eval(shipped, [case()], [exact(), exact(), contains()])
    assert [g.grader for g in report.cases[0].grades] == ["exact", "exact#2", "contains"]


async def test_concurrency_caps_cases_at_once():
    running = peak = 0

    async def agent(question):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return "shipped"

    cases = [case(f"c{i}") for i in range(6)]
    report = await run_eval(agent, cases, [exact()], concurrency=2)
    assert peak == 2 and [c.case_id for c in report.cases] == [f"c{i}" for i in range(6)]


async def test_cancellation_is_not_a_case_error():
    async def hang(question):
        await asyncio.sleep(10)

    task = asyncio.ensure_future(run_eval(hang, [case()], [exact()]))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cases_can_come_from_a_file(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"id": "a", "input": "q", "expected": "shipped"}) + "\n", encoding="utf-8"
    )
    report = await run_eval(shipped, str(path), [exact()])
    assert report.n_passed == 1


def test_run_eval_sync_runs_in_sync_code():
    assert run_eval_sync(shipped, [case()], [exact()]).n_passed == 1


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"concurrency": 0}, "concurrency"),
        ({"concurrency": True}, "concurrency"),
        ({"timeout_s": 0}, "timeout_s"),
        ({"timeout_s": float("inf")}, "timeout_s"),
        ({"pass_at": 1.5}, "pass_at"),
        ({"executor": object()}, "must be an Executor"),
    ],
)
async def test_bad_options_are_refused(options, message):
    with pytest.raises(ConfigError, match=message):
        await run_eval(shipped, [case()], [exact()], **options)


@pytest.mark.parametrize(
    ("cases", "graders", "message"),
    [
        ([], [exact()], "at least one case"),
        ([case("a"), case("a")], [exact()], "duplicate case ids"),
        (["a"], [exact()], "EvalCase objects"),
        ([case()], [], "at least one grader"),
        ([case()], {"x": "not callable"}, "must be callable"),
    ],
)
async def test_bad_cases_or_graders_are_refused(cases, graders, message):
    with pytest.raises(ConfigError, match=message):
        await run_eval(shipped, cases, graders)


def test_collector_keeps_open_runs_only_and_caps_them(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(_collect, "MAX_EVENTS", 2)
    collector = Collector()
    collector.emit(Event("tool.started", run_id="r1", data={"tool": "a"}))
    collector.open("r1")
    for _ in range(4):
        collector.emit(Event("tool.started", run_id="r1", data={"tool": "a"}))
    assert len(collector.close("r1")) == 2
    assert caplog.text.count("the rest are dropped") == 1
    assert collector.close("r1") == ()


def test_case_cost_takes_the_larger_of_ledger_and_events():
    events = (
        Event("llm.finished", run_id="r", data={"cost_usd": 0.25}),
        Event("embedding.finished", run_id="r", data={"cost_usd": None}),
        Event("context.compressed", run_id="r", data={"summary_cost_usd": 0.5}),
        Event("rerank.finished", run_id="r", data={"cost_usd": True}),
    )

    class Fixed:
        def total_usd(self, run_id=None, *, thread_id=None):
            return 2.0

    assert case_cost(None, "r", events) == 0.75
    assert case_cost(Fixed(), "r", events) == 2.0


async def test_a_case_without_expected_fails_closed():
    report = await run_eval(shipped, [EvalCase(id="x", input="q")], [exact()])
    assert not report.cases[0].passed and report.cases[0].score == 0.0


class GraphBubbleUp(Exception):
    """Named like LangGraph's interrupt base, so it counts as control flow."""


async def test_control_flow_from_a_target_or_grader_propagates_unwrapped():
    def interrupt(question):
        raise GraphBubbleUp

    async def interrupting(c, r):
        raise GraphBubbleUp

    with pytest.raises(GraphBubbleUp):
        await run_eval(interrupt, [case()], [exact()])
    with pytest.raises(GraphBubbleUp):
        await run_eval(shipped, [case()], [interrupting])
    for wrapped in (code_check(interrupting), all_of(interrupting), LLMJudge(Interrupting(), "r")):
        with pytest.raises(GraphBubbleUp):
            await wrapped(case(), EvalRun(case_id="c1", output="x"))


async def test_a_bad_expected_pattern_is_ungraded(caplog):
    caplog.set_level(logging.WARNING)
    assert await regex()(case(expected="("), EvalRun(case_id="c1", output="x")) is None
    assert "regex grader failed" in caplog.text


class Interrupting:
    async def generate(self, request):
        raise GraphBubbleUp


def test_the_package_lists_its_names():
    import agent_adapter.evals as evals

    assert dir(evals) == sorted(evals.__all__)
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        evals.nope  # noqa: B018


async def test_max_cost_without_a_cost_is_ungraded():
    class Bare:
        output = "x"

    assert await max_cost(1)(case(), Bare()) is None


async def test_an_interrupted_case_still_finishes_its_run():
    def interrupt(question):
        raise GraphBubbleUp

    sink = RecordingSink()
    with pytest.raises(GraphBubbleUp):
        await run_eval(
            interrupt, [case()], [exact()], executor=Executor(shop_registry(), sink=sink)
        )
    assert [e.type for e in sink.events] == ["run.started", "run.finished"]
    assert sink.events[-1].data["ok"] is False


async def test_all_of_with_an_erroring_or_ungraded_part_fails_the_case():
    def boom(c, r):
        raise RuntimeError("bug")

    erroring = await run_eval(shipped, [case()], [all_of(contains(), code_check(boom))])
    ungraded = await run_eval(shipped, [case(expected=None)], [all_of(contains(), max_latency(10))])
    for report in (erroring, ungraded):
        grade = report.cases[0].grades[0]
        assert grade.passed is False and not report.cases[0].passed
    assert "all_of[1] failed to grade" in erroring.cases[0].grades[0].notes
    assert "all_of[0] not graded" in ungraded.cases[0].grades[0].notes


async def test_ungraded_and_failed_grades_pull_the_case_score_down():
    async def high_but_failed(c, r):
        return Score(value=1.0, passed=False)

    async def high(c, r):
        return Score(value=0.9)

    no_expected = await run_eval(shipped, [case(expected=None)], [contains(), max_latency(10)])
    assert no_expected.cases[0].score == 0.5 and no_expected.n_passed == 0
    capped = await run_eval(shipped, [case()], [high], pass_at=0.95)
    failed = await run_eval(shipped, [case()], [high_but_failed], pass_at=0.5)
    assert capped.cases[0].score == pytest.approx(0.9) and failed.cases[0].score == 0.5


async def test_a_sync_grader_is_run_as_a_check():
    def sync(c, r):
        return Score(value=1.0, passed=True, notes="sync ok")

    result = (await run_eval(shipped, [case()], [sync])).cases[0]
    assert result.passed and result.grades[0].grader == "sync"
    assert result.grades[0].notes == "sync ok"


async def test_unobserved_targets_leave_cost_and_tool_graders_ungraded():
    registry = shop_registry()
    own = Executor(registry)

    async def blind(question):  # calls a tool on its own executor, outside the case run
        await own.run("order_status", {"order_id": "1"})
        return "shipped"

    report = await run_eval(blind, [case()], [tools_called(forbid=["order_status"]), max_cost(1)])
    result = report.cases[0]
    assert [g.score for g in result.grades] == [None, None] and not result.passed
    assert result.measured is False


async def test_a_target_that_uses_the_case_run_is_measured():
    registry = shop_registry()
    executor = Executor(registry)

    async def seen(question, ctx):
        return await executor.run("order_status", {"order_id": "1042"}, ctx)

    result = (await run_eval(seen, [case()], [max_cost(1)], executor=executor)).cases[0]
    assert result.passed and result.measured


async def test_a_ledger_that_fails_does_not_abort_the_eval(caplog):
    caplog.set_level(logging.WARNING)

    class Broken:
        def record(self, entry):
            pass

        def total_usd(self, run_id=None, *, thread_id=None):
            raise OSError("db down")

        def entries(self, run_id=None, *, thread_id=None):
            return []

    executor = Executor(shop_registry(), ledger=Broken())
    report = await run_eval(
        shipped, [case("a"), case("b")], [exact(), max_cost(1)], executor=executor
    )
    assert report.n_cases == 2 and not any(c.passed for c in report.cases)
    assert all(c.grades[0].passed and c.grades[1].score is None for c in report.cases)
    assert "cannot measure it" in caplog.text and "db down" not in caplog.text


async def test_lone_surrogates_never_break_the_saved_report(tmp_path):
    def broken_text(question):
        return "bad \ud800 text"

    report = await run_eval(
        broken_text, [case("s\udc00", input="q \ud800")], [contains()], save_to=tmp_path
    )
    result = report.cases[0]
    assert result.output == "bad ? text" and result.case_id == "s?" and result.input == "q ?"
    assert EvalReport.from_json(report.to_json()) == report
    assert len(list(tmp_path.glob("*.json"))) == 1


async def test_control_flow_from_the_ledger_propagates():
    class Interrupting:
        def record(self, entry):
            pass

        def total_usd(self, run_id=None, *, thread_id=None):
            raise GraphBubbleUp

        def entries(self, run_id=None, *, thread_id=None):
            return []

    with pytest.raises(GraphBubbleUp):
        await run_eval(shipped, [case()], [exact()], executor=Executor(ledger=Interrupting()))
