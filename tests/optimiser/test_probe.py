import asyncio

import pytest

from agent_adapter.cache.memory import MemoryCache
from agent_adapter.cost.ledger import Ledger
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.optimiser.model import Candidate, CaseRun, Edit, HarnessTree, ProbeReport, Score
from agent_adapter.optimiser.probe import contrast, probe
from tests.optimiser.fakes import CALL_USD, FakeHost, ToyAgent


def instrumented(agent, ledger, cache=None):
    return InstrumentedModel(agent.port, ledger=ledger, cache=cache, namespace="t")


async def test_probe_runs_every_case_once_and_judges_it():
    host, agent, ledger = FakeHost(), ToyAgent(), Ledger()
    tree = host.surface().with_edits({"prompt/system": "cite the section"})
    report = await probe(
        host,
        tree,
        "know1",
        model=instrumented(agent, ledger),
        run_prefix="opt_x",
        ledger=ledger,
        concurrency=2,
    )
    assert [r.case_id for r in report.runs] == ["q1", "q2", "q3"]
    assert {r.label for r in report.runs} == {"know1"}
    assert sorted(run_id for run_id, _ in host.runs) == [
        "opt_x/know1/q1",
        "opt_x/know1/q2",
        "opt_x/know1/q3",
    ]
    assert (report.mean_score, report.n_passed, report.n_graded, report.n_delivered) == (
        1.0,
        3,
        3,
        3,
    )
    assert report.cost_usd == pytest.approx(3 * CALL_USD)
    assert report.runs[0].metrics["cost_usd"] == pytest.approx(CALL_USD)
    assert report.runs[0].judge_notes == "cites it"


async def test_second_base_probe_is_free_through_the_response_cache():
    host, agent, ledger = FakeHost(), ToyAgent(), Ledger()
    model = instrumented(agent, ledger, MemoryCache())
    first = await probe(
        host, host.surface(), "base", model=model, run_prefix="o", ledger=ledger, concurrency=4
    )
    again = await probe(
        host, host.surface(), "kb1", model=model, run_prefix="o", ledger=ledger, concurrency=4
    )
    assert first.cost_usd == pytest.approx(3 * CALL_USD)
    assert again.cost_usd == 0.0
    assert len(agent.requests) == 3
    assert [r.output for r in again.runs] == [r.output for r in first.runs]


async def test_judge_failures_and_host_errors_never_raise():
    async def judge(case, run):
        if case.id == "q1":
            raise RuntimeError("judge down")
        if case.id == "q2":
            return None
        return Score(value=0.5, passed=False)

    class Exploding(FakeHost):
        async def run(self, tree, case, *, run_id, model):
            if case.id == "q3":
                raise ValueError("host bug")
            return await super().run(tree, case, run_id=run_id, model=model)

    host, ledger = Exploding(judge=judge), Ledger()
    report = await probe(
        host,
        host.surface(),
        "base",
        model=instrumented(ToyAgent(), ledger),
        run_prefix="o",
        ledger=ledger,
        concurrency=1,
    )
    q1, q2, q3 = report.runs
    assert (q1.score, q2.score) == (None, None)
    assert q3.ok is False and q3.error == "internal-error: ValueError"
    assert q3.score == 0.5  # the judge still sees the failed run
    assert (report.n_graded, report.mean_score, report.n_delivered) == (1, 0.5, 2)


async def test_probe_without_a_judge_has_no_scores():
    host, ledger = FakeHost(judge=None), Ledger()
    report = await probe(
        host,
        host.surface(),
        "base",
        model=instrumented(ToyAgent(), ledger),
        run_prefix="o",
        ledger=ledger,
        concurrency=3,
    )
    assert report.mean_score is None and report.n_graded == 0


async def test_probe_lets_cancellation_through():
    class Slow(FakeHost):
        async def run(self, tree, case, *, run_id, model):
            await asyncio.sleep(10)

    ledger = Ledger()
    task = asyncio.ensure_future(
        probe(
            Slow(),
            HarnessTree(files={}),
            "base",
            model=instrumented(ToyAgent(), ledger),
            run_prefix="o",
            ledger=ledger,
            concurrency=2,
        )
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def case_run(case_id, label, *, delivered=True, score=None, metrics=None, transcript="t"):
    return CaseRun(
        case_id=case_id,
        label=label,
        ok=True,
        delivered=delivered,
        output="o",
        transcript=transcript,
        metrics=metrics or {},
        score=score,
    )


def rep(label, *runs):
    return ProbeReport(
        label=label,
        runs=list(runs),
        mean_score=None,
        n_passed=0,
        n_graded=0,
        n_delivered=0,
        cost_usd=0.0,
    )


def cand(*paths):
    return Candidate(
        label="c",
        dim="procedure",
        parent_label="base",
        tree=HarnessTree(files={}),
        edits=tuple(Edit(path=p, content="x", reason="r") for p in paths),
    )


def test_contrast_reports_deltas_delivery_and_a_bounded_diff():
    ref = rep(
        "base",
        case_run("q1", "base", score=0.2, metrics={"steps": 4}, transcript="a\nb"),
        case_run("q2", "base", score=None),
    )
    new = rep(
        "c",
        case_run("q1", "c", score=0.7, metrics={"steps": 2}, transcript="a\nc" * 2000),
        case_run("q2", "c", delivered=False, score=0.1),
        case_run("q9", "c"),
    )
    rows = contrast(ref, new, cand("prompt/system"), diff_chars=300)
    assert [r.case_id for r in rows] == ["q1", "q2"]
    first, second = rows
    assert first.score_delta == pytest.approx(0.5) and second.score_delta is None
    assert first.metric_deltas == {"steps": -2}
    assert second.delivered == (True, False)
    assert first.edit_exercised is None  # prompt edits are not observable
    assert len(first.transcript_diff) <= 300 and first.transcript_diff.startswith("--- base")


@pytest.mark.parametrize(
    ("paths", "ref_metrics", "cand_metrics", "expected"),
    [
        (
            ["tool/search/description"],
            {"tool_calls": 0},
            {"tool_calls": 1, "tool_calls.search": 1},
            True,
        ),
        (
            ["tool/search/description"],
            {"tool_calls": 2, "tool_calls.search": 2},
            {"tool_calls": 0},
            True,
        ),
        (["tool/search/prompt_hint"], {"tool_calls": 0}, {"tool_calls": 0}, False),
        (["tool/search/prompt_hint"], {}, {}, None),
        (["hook/budget/config.json"], {"hook_fired.budget": 0}, {"hook_fired.budget": 1}, True),
        (["hook/budget/config.json"], {"hook_fired.budget": 0}, {"hook_fired.budget": 0}, False),
        (["hook/budget/config.json"], {}, {}, None),
        (["tool/search/prompt_hint", "prompt/system"], {"tool_calls": 0}, {"tool_calls": 0}, None),
        (
            ["tool/search/prompt_hint", "prompt/system"],
            {"tool_calls": 0},
            {"tool_calls.search": 1},
            True,
        ),
    ],
)
def test_edit_exercised(paths, ref_metrics, cand_metrics, expected):
    ref = rep("base", case_run("q1", "base", metrics=ref_metrics))
    new = rep("c", case_run("q1", "c", metrics=cand_metrics))
    assert contrast(ref, new, cand(*paths))[0].edit_exercised is expected


async def test_a_judge_returning_something_else_counts_as_a_failure():
    async def judge(case, run):
        return 0.5

    host, ledger = FakeHost(judge=judge), Ledger()
    report = await probe(
        host,
        host.surface(),
        "base",
        model=instrumented(ToyAgent(), ledger),
        run_prefix="o",
        ledger=ledger,
        concurrency=3,
    )
    assert report.n_graded == 0


async def test_a_case_that_hangs_times_out_without_stopping_the_others():
    class Slow(FakeHost):
        async def run(self, tree, case, *, run_id, model):
            if case.id == "q2":
                await asyncio.sleep(10)
            return await super().run(tree, case, run_id=run_id, model=model)

    host, ledger = Slow(), Ledger()
    report = await probe(
        host,
        host.surface(),
        "base",
        model=instrumented(ToyAgent(), ledger),
        run_prefix="o",
        ledger=ledger,
        concurrency=3,
        case_timeout_s=0.02,
    )
    q1, q2, q3 = report.runs
    assert q1.ok and q3.ok
    assert not q2.ok and q2.error == "case-timeout: case q2 took longer than 0.02 s"
