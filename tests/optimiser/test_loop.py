"""The optimiser loop end to end: rounds, compose, budgets, resume and lessons."""

import json

import pytest

from agent_adapter.cache.memory import MemoryCache
from agent_adapter.cost.ledger import Ledger
from agent_adapter.optimiser import OptimiserBudget, _budget
from tests.optimiser.fakes import (
    CALL_USD,
    FakeHost,
    keep_base,
    proposal,
)
from tests.optimiser.loop_support import CITE, HINT, LESSON, run


async def test_parallel_k1_adopts_the_helpful_edit_and_writes_every_output(tmp_path):
    result, proposer, agent, host = await run(
        tmp_path, {"proc1": proposal(HINT), "know1": proposal(CITE), "reflect": LESSON}
    )
    assert result.chosen == "agg1"
    assert result.finals == ["agg1"]
    assert result.base.mean_score == 0.0 and result.probes["agg1"].mean_score == 1.0
    assert result.candidates["agg1"].tree.files["prompt/system"] == CITE[1]
    assert "tool/search/prompt_hint" not in result.candidates["agg1"].tree.files  # no gain
    # budget identity: (1 + 3k) * N case runs, 2k proposer calls
    assert result.budget.case_runs == 4 * 3 and len(host.runs) == 12
    assert (result.budget.proposer_calls, result.budget.merge_calls) == (2, 0)
    assert proposer.stages() == ["propose:proc1", "propose:know1", "reflect"]
    assert len(agent.requests) == 9  # agg1 equals know1's tree: its probe is served from cache
    assert result.probes["agg1"].cost_usd == 0.0
    assert result.budget.usd == pytest.approx(9 * CALL_USD + 3 * CALL_USD)
    assert result.ledger_summary.by_stage["propose:proc1"].calls == 1
    kinds = {(d.kind, d.path) for d in result.decisions}
    assert ("kept", "prompt/system") in kinds and ("dropped", "tool/search/prompt_hint") in kinds
    assert result.patch.startswith("--- a/prompt/system\n+++ b/prompt/system\n")
    assert result.errors == []
    out = tmp_path / "out"
    assert (out / "harness.patch").read_text("utf-8") == result.patch
    assert "Chosen: **agg1**" in (out / "report.md").read_text("utf-8")
    after = json.loads((out / "harness.after.json").read_text("utf-8"))
    assert after["label"] == "agg1" and after["files"]["prompt/system"] == CITE[1]
    saved = json.loads((out / "result.json").read_text("utf-8"))
    assert saved["chosen"] == "agg1" and saved["base"]["runs"][0]["transcript"] == "runs/base/q1.md"
    assert (out / "runs" / "agg1" / "q2.md").read_text("utf-8").startswith("# agg1 / q2")
    lessons = json.loads((tmp_path / "lessons.json").read_text("utf-8"))["lessons"]
    assert [x["cue"] for x in lessons] == [LESSON["cue"]] and lessons[0]["adopted"]
    assert [x.cue for x in result.lessons_added] == [LESSON["cue"]]


async def test_keep_base_yields_an_empty_patch(tmp_path):
    result, proposer, agent, _ = await run(tmp_path, {"proc1": keep_base(), "know1": keep_base()})
    assert result.chosen is None and result.patch == ""
    assert result.budget.case_runs == 12 and result.budget.proposer_calls == 2
    assert len(agent.requests) == 3  # every other probe was a cache hit
    assert "reflect" not in proposer.stages()  # nothing proposed, nothing to reflect on
    after = json.loads((tmp_path / "out" / "harness.after.json").read_text("utf-8"))
    assert after["label"] == "base"
    assert (tmp_path / "out" / "harness.patch").read_text("utf-8") == ""
    assert "Chosen: **KEEP_BASE**" in result.report_md


async def test_parallel_k2_budget_identity_and_finals(tmp_path):
    result, proposer, _, host = await run(
        tmp_path,
        {
            "proc1": proposal(HINT),
            "know1": proposal(CITE),
            "know2": keep_base(),
            "proc2": keep_base(),
        },
        k=2,
    )
    assert result.budget.case_runs == (1 + 3 * 2) * 3 == len(host.runs)
    assert result.budget.proposer_calls == 4
    assert result.finals == ["agg1", "agg2"]
    assert result.chosen in {"agg1", "agg2"}
    seeds = {r.seed for r in proposer.requests if r.stage.startswith("propose:")}
    assert len(seeds) == 4  # one deterministic seed per proposer


async def test_sequential_rounds_build_on_the_checkpoint(tmp_path):
    result, proposer, _, host = await run(
        tmp_path,
        {"know1": proposal(CITE), "proc1": keep_base(), "proc2": keep_base(), "know2": keep_base()},
        k=2,
        mode="sequential",
    )
    assert result.finals == ["c1", "c2"]
    assert result.budget.case_runs == 21 == len(host.runs)
    assert result.budget.proposer_calls == 4
    assert result.candidates["proc2"].parent_label == "c1"
    round2 = next(r for r in proposer.requests if r.stage == "propose:know2")
    assert "Round 1 -> c1" in round2.messages[1].content
    assert CITE[1] in round2.messages[1].content  # the checkpoint is the current harness
    assert "checkpoint" in round2.messages[0].content
    assert result.chosen in {"c1", "c2"}


async def test_llm_compose_merges_a_conflict_once(tmp_path):
    script = {
        "proc1": keep_base(),
        "proc2": keep_base(),
        "know1": proposal(("prompt/system", "Cite the section, version A.")),
        "know2": proposal(("prompt/system", "Cite the section, version B.")),
        "agg1": {
            "merges": [
                {
                    "path": "prompt/system",
                    "content": "Cite the section (A+B).",
                    "decision": "kept both",
                }
            ]
        },
    }
    result, proposer, _, _ = await run(tmp_path, script, k=2, compose_mode="llm")
    assert result.budget.merge_calls == 1 and result.budget.proposer_calls == 4
    assert proposer.stages().count("merge:agg1") == 1 and "merge:agg2" not in proposer.stages()
    assert result.candidates["agg1"].tree.files["prompt/system"] == "Cite the section (A+B)."
    assert any(d.evidence == "[agg1] merged by LLM: kept both" for d in result.decisions)


async def test_greedy_compose_takes_one_side_of_a_conflict(tmp_path):
    script = {
        "proc1": keep_base(),
        "proc2": keep_base(),
        "know1": proposal(("prompt/system", "Cite the section, version A.")),
        "know2": proposal(("prompt/system", "Cite the section, version B.")),
    }
    result, proposer, _, _ = await run(tmp_path, script, k=2)
    assert result.budget.merge_calls == 0
    assert not any(s.startswith("merge:") for s in proposer.stages())
    assert any(d.kind == "revised" and "conflict: took" in d.evidence for d in result.decisions)


async def test_unsafe_candidate_is_never_admitted(tmp_path):
    empty = ("prompt/system", "EMPTY but cite")
    result, *_ = await run(tmp_path, {"proc1": keep_base(), "know1": proposal(empty)})
    assert result.chosen is None
    assert any(
        d.kind == "dropped" and "unsafe" in d.evidence and d.path == "prompt/system"
        for d in result.decisions
    )
    assert result.candidates["agg1"].edits == ()


async def test_retry_then_fall_back_is_reported(tmp_path):
    bad = proposal(("prompt/system", "x" * 20_000))
    result, proposer, *_ = await run(tmp_path, {"proc1": keep_base(), "know1": [bad, bad]})
    know1 = result.candidates["know1"]
    assert know1.fell_back and know1.attempts == 2
    assert result.budget.proposer_calls == 2  # retries count as attempts, not calls
    assert proposer.stages().count("propose:know1") == 2
    assert result.errors[0].startswith("know1: fell back to base:")


async def test_case_run_budget_stops_with_a_partial_report(tmp_path):
    result, _, _, host = await run(
        tmp_path,
        {"proc1": proposal(HINT), "know1": proposal(CITE)},
        budget=OptimiserBudget(max_case_runs=5),
    )
    assert result.budget.stopped == "max_case_runs"
    assert result.budget.case_runs == 3 == len(host.runs)
    assert result.chosen is None and result.patch == ""
    assert result.errors[-1] == "stopped early: max_case_runs"
    report = (tmp_path / "out" / "report.md").read_text("utf-8")
    assert "Stopped early: max_case_runs" in report
    assert (tmp_path / "out" / "result.json").exists()


async def test_usd_budget_hook_stops_the_run(tmp_path):
    result, _, agent, _ = await run(
        tmp_path,
        {"proc1": proposal(HINT), "know1": proposal(CITE)},
        budget=OptimiserBudget(max_usd=CALL_USD * 1.5),
        rollout_concurrency=1,
    )
    assert result.budget.stopped == "max_usd"
    assert len(agent.requests) == 2  # the BudgetHook refused the third agent call
    assert result.base.runs[2].error.startswith("budget-exceeded")
    assert result.budget.proposer_calls == 0


async def test_wall_clock_budget(tmp_path, monkeypatch):
    ticks = iter(range(0, 10_000, 100))
    monkeypatch.setattr(_budget, "_now", lambda: float(next(ticks)))
    result, *_ = await run(tmp_path, {}, budget=OptimiserBudget(max_wall_s=50))
    assert result.budget.stopped == "max_wall_s"
    assert result.base.runs == [] and result.budget.case_runs == 0


async def test_a_stopped_run_resumes_without_repeating_finished_stages(tmp_path):
    script = {"proc1": proposal(HINT), "know1": proposal(CITE)}
    cache = MemoryCache()
    first, *_ = await run(tmp_path, script, budget=OptimiserBudget(max_case_runs=6), cache=cache)
    assert first.budget.stopped == "max_case_runs" and first.budget.case_runs == 6
    second, proposer2, _, host2 = await run(tmp_path, script, cache=cache)
    assert second.budget.stopped == "" and second.chosen == "agg1"
    assert not any(s.startswith("propose:") for s in proposer2.stages())
    assert second.budget.case_runs == 12 and second.budget.proposer_calls == 2
    assert len(host2.runs) == 6  # only know1 and agg1 ran this time
    assert (tmp_path / "out" / "state.json").exists()
    third, proposer3, _, host3 = await run(tmp_path, script, cache=cache)
    assert third.chosen == "agg1" and host3.runs == [] and proposer3.requests == []


async def test_changed_inputs_start_afresh(tmp_path):
    await run(tmp_path, {"proc1": keep_base(), "know1": keep_base()})
    _, proposer, *_ = await run(tmp_path, {"proc1": keep_base(), "know1": keep_base()}, seed=1)
    assert proposer.stages().count("propose:proc1") == 1


async def test_lessons_feed_later_runs_and_no_entry_is_not_stored(tmp_path):
    no_entry = {**LESSON, "no_entry": True}
    first, *_ = await run(
        tmp_path, {"proc1": keep_base(), "know1": proposal(CITE), "reflect": no_entry}
    )
    assert first.lessons_added == [] and not (tmp_path / "lessons.json").exists()
    await run(
        tmp_path,
        {"proc1": keep_base(), "know1": proposal(CITE), "reflect": LESSON},
        out_dir=str(tmp_path / "o2"),
    )
    _, proposer, *_ = await run(tmp_path, {}, out_dir=str(tmp_path / "o3"))
    evidence = next(r for r in proposer.requests if r.stage == "propose:know1").messages[1].content
    assert "# EARLIER RUNS IN THIS AREA" in evidence and LESSON["cue"] in evidence
    _, proposer, *_ = await run(tmp_path, {}, out_dir=str(tmp_path / "o4"), lessons_path="")
    assert "EARLIER RUNS" not in proposer.requests[0].messages[1].content


async def test_without_a_judge_a_blind_selector_decides(tmp_path):
    script = {"proc1": keep_base(), "know1": proposal(CITE), "select": {"why": "w", "choice": "Z"}}
    result, proposer, *_ = await run(tmp_path, script, host=FakeHost(judge=None))
    assert "select" in proposer.stages()
    assert result.chosen is None  # "Z" is not an offered letter: neutral fallback


async def test_the_callers_ledger_receives_every_entry(tmp_path):
    ledger = Ledger()
    result, *_ = await run(tmp_path, {"proc1": keep_base(), "know1": keep_base()}, ledger=ledger)
    assert ledger.total_usd() == pytest.approx(result.budget.usd)
    assert len(ledger.entries()) == sum(s.calls for s in result.ledger_summary.by_kind.values())
