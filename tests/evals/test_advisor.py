"""`advise`: per-step advice from evals and the ledger, budgets, reports and the example."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import LLMRequest
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import LedgerEntry
from agent_adapter.cost.ledger import Ledger
from agent_adapter.cost.prices import Prices
from agent_adapter.evals import EvalCase, LLMJudge, ToolLoopTarget
from agent_adapter.evals.advisor import AdvisorReport, advise, advise_sync
from agent_adapter.evals.graders import code_check, contains
from agent_adapter.exec.executor import Executor
from agent_adapter.models.callable import CallableModel

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "evals_cost_advisor.py"
TABLE = {
    "version": "t",
    "models": {
        "big": {"provider": "p", "input": 10.0, "output": 10.0},
        "mid": {"provider": "p", "input": 2.0, "output": 2.0},
        "tiny": {"provider": "p", "input": 0.1, "output": 0.1},
        "huge": {"provider": "p", "input": 100.0, "output": 100.0},
    },
}
CASES = [EvalCase(id=i, input="ping", expected="pong") for i in ("a", "b")]


def scripted(name: str, *, good: bool = True, tokens: int = 100) -> CallableModel:
    def answer(request: LLMRequest) -> dict[str, Any]:
        text = "pong" if good else "nope"
        return {"text": text, "usage": {"input_tokens": tokens, "output_tokens": tokens}}

    return CallableModel(answer, provider="p", model=name)


def loop_target(model: Any) -> ToolLoopTarget:
    return ToolLoopTarget(model, Registry())


def test_the_example_suggests_a_cheaper_model_per_node(capsys: pytest.CaptureFixture[str]):
    found = runpy.run_path(str(EXAMPLE), run_name="example")["main"]()
    assert found["mapping"] == {"node:reply": "medium", "node:triage": "small"}
    assert found["combined_cost"] < found["baseline_cost"]
    assert 0 < found["spent"] <= 0.50
    assert "ticket 19" in found["reply"]
    assert "# Cost advice" in capsys.readouterr().out


async def test_a_tool_loop_step_gets_the_cheapest_model_that_keeps_the_score():
    report = await advise(
        loop_target,
        CASES,
        [contains()],
        candidates={"big": scripted("big"), "mid": scripted("mid"), "tiny": scripted("tiny")},
        baseline="big",
        prices=TABLE,
    )
    assert report.mapping == {"stage:tool_loop": "tiny"}
    (advice,) = report.steps
    assert advice.baseline.calls == 2 and advice.cost_delta_usd < 0
    assert advice.run_cost_delta_usd == pytest.approx(advice.cost_delta_usd)
    assert report.combined is not None and report.combined.model == "tiny"
    assert [t.model for t in report.trials] == ["tiny", "mid"]  # cheapest list price first
    assert report.stopped == "" and report.max_spend_usd is None


async def test_no_advice_when_every_cheaper_model_loses_the_score():
    report = await advise(
        loop_target,
        CASES,
        [contains()],
        candidates={"big": scripted("big"), "tiny": scripted("tiny", good=False)},
        baseline="big",
        prices=Prices.empty("t").merged(TABLE),
    )
    assert report.mapping == {} and report.combined is None
    assert report.steps[0].reason == "no cheaper candidate keeps the score"
    assert report.trials[0].keeps_score is False


async def test_a_dearer_model_that_keeps_the_score_is_not_advised():
    report = await advise(
        loop_target,
        CASES,
        [contains()],
        candidates={"mid": scripted("mid"), "big": scripted("big")},
        baseline="mid",
        prices=TABLE,
    )
    assert report.mapping == {} and report.trials[0].keeps_score


async def test_per_step_false_swaps_every_step_at_once():
    report = await advise(
        loop_target,
        CASES,
        [contains()],
        candidates={"big": scripted("big"), "tiny": scripted("tiny")},
        baseline="big",
        prices=TABLE,
        per_step=False,
    )
    assert report.mapping == {"*": "tiny"}
    assert report.steps[0].baseline.calls == 2


async def test_a_baseline_that_scores_zero_gets_no_advice():
    report = await advise(
        loop_target,
        CASES,
        [contains()],
        candidates={"big": scripted("big", good=False), "tiny": scripted("tiny", good=False)},
        baseline="big",
        prices=TABLE,
    )
    assert report.mapping == {} and "scores 0" in report.steps[0].reason


async def test_unpriced_calls_are_never_advised():
    candidates = {"big": scripted("big"), "free": scripted("unknown-model")}
    report = await advise(
        loop_target, CASES, [contains()], candidates=candidates, baseline="big", prices=TABLE
    )
    assert report.mapping == {} and report.trials[0].steps["stage:tool_loop"].unpriced == 2
    flipped = await advise(
        loop_target, CASES, [contains()], candidates=candidates, baseline="free", prices=TABLE
    )
    assert flipped.steps[0].reason == "baseline calls are unpriced"


async def test_a_free_baseline_has_nothing_to_save():
    report = await advise(
        loop_target,
        CASES,
        [contains()],
        candidates={"big": scripted("big", tokens=0), "tiny": scripted("tiny", tokens=0)},
        baseline="big",
        prices=TABLE,
    )
    assert report.mapping == {} and report.steps[0].reason == "baseline costs 0: nothing to save"


async def test_the_spend_cap_stops_early_and_is_never_exceeded():
    # a hold is ~6.5K tokens (request bytes, margin, 4096 out): ~$0.065 on big, ~$0.65 on huge
    report = await advise(
        loop_target,
        CASES,
        [contains()],
        candidates={
            "big": scripted("big"),
            "mid": scripted("mid"),
            "tiny": scripted("tiny"),
            "huge": scripted("huge"),
        },
        baseline="big",
        prices=TABLE,
        max_spend=0.2,
        concurrency=1,
    )
    assert report.stopped == "budget"
    assert [(t.model, t.complete) for t in report.trials] == [
        ("tiny", True),
        ("mid", True),
        ("huge", False),
    ]
    assert report.mapping == {"stage:tool_loop": "tiny"}
    assert 0 < report.spent_usd <= 0.2
    assert "Stopped early: budget." in report.to_markdown()


async def test_a_cap_too_small_for_the_baseline_gives_no_baseline():
    report = await advise(
        loop_target,
        CASES,
        [contains()],
        candidates={"big": scripted("big"), "tiny": scripted("tiny")},
        baseline="big",
        prices=TABLE,
        max_spend=0.000001,
    )
    assert report.baseline_trial is None and report.stopped == "budget"
    assert report.spent_usd == 0 and "No baseline run (budget)" in report.to_markdown()


async def test_the_judge_model_is_not_a_step():
    judge = CallableModel(
        lambda r: json.dumps({"value": 1.0, "passed": True}), provider="p", model="big"
    )
    report = await advise(
        loop_target,
        CASES,
        {"answer": contains(), "judge": LLMJudge(judge, "Says pong.")},
        candidates={"big": scripted("big"), "tiny": scripted("tiny")},
        baseline="big",
        executor=Executor(Registry(), ledger=Ledger(prices=Prices.empty("t").merged(TABLE))),
    )
    assert [a.step for a in report.steps] == ["stage:tool_loop"]  # prices from the executor
    assert report.mapping == {"stage:tool_loop": "tiny"}


async def test_judge_calls_count_against_the_cap():
    calls: list[int | None] = []

    def grade(request: LLMRequest) -> dict[str, Any]:
        calls.append(request.max_output_tokens)
        usage = {"input_tokens": 1000, "output_tokens": 1000}
        return {"text": json.dumps({"value": 1.0, "passed": True}), "usage": usage}

    judge = CallableModel(grade, provider="p", model="big")
    report = await advise(
        loop_target,
        CASES,
        [contains(), LLMJudge(judge, "Says pong.")],
        candidates={"big": scripted("big"), "tiny": scripted("tiny")},
        baseline="big",
        prices=TABLE,
        max_spend=0.2,
        max_output_tokens=256,
        concurrency=1,
    )
    assert calls and set(calls) == {256}  # judge calls are limited and held like the rest
    assert report.spent_usd >= 0.02 * len(calls)  # each judge call is $0.02 of the spend
    assert report.spent_usd <= 0.2


async def test_an_unpriced_judge_is_refused_under_a_cap():
    judge = CallableModel(lambda r: "{}", provider="p", model="no-price")
    with pytest.raises(ConfigError, match="judge of grader 1"):
        await advise(
            loop_target,
            CASES,
            [contains(), LLMJudge(judge, "Says pong.")],
            candidates={"big": scripted("big")},
            baseline="big",
            prices=TABLE,
            max_spend=1.0,
        )


class Flow2:
    """Two stages in one target, so the combined run is verified."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.name = "two_stages"

    async def run_case(self, case: EvalCase, *, executor: Executor, ctx: Any) -> str:
        llm = executor.model(self.model).bind(ctx)
        first = await llm.generate(_ask("draft"))
        second = await llm.generate(_ask("check"))
        return f"{first.text} {second.text}"


def _ask(stage: str) -> LLMRequest:
    from agent_adapter.core.messages import Message

    return LLMRequest(messages=(Message(role="user", content="ping"),), stage=stage)


@pytest.mark.parametrize("verify", [True, False])
async def test_two_steps_are_verified_together_unless_told_not_to(verify: bool):
    report = await advise(
        Flow2,
        CASES,
        [code_check(lambda case, run: run.output == "pong pong")],
        candidates={"big": scripted("big"), "tiny": scripted("tiny")},
        baseline="big",
        prices=TABLE,
        verify=verify,
    )
    assert (report.combined is not None) is verify
    assert report.verified is verify
    assert report.advised == {"stage:check": "tiny", "stage:draft": "tiny"}
    if not verify:
        assert report.mapping == {} and report.note == "not verified: verify=False"
    if report.combined is not None:
        assert report.combined.keeps_score and report.combined.step == "combined"
        assert report.combined.cost_usd < report.baseline_trial.cost_usd  # type: ignore[union-attr]


class Picky:
    """Each stage alone may go cheap, but not both: the combined run loses the score."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.name = "picky"

    async def run_case(self, case: EvalCase, *, executor: Executor, ctx: Any) -> str:
        llm = executor.model(self.model).bind(ctx)
        first = await llm.generate(_ask("draft"))
        second = await llm.generate(_ask("check"))
        return f"{first.model} {second.model}"


async def test_a_failed_combined_run_empties_the_mapping():
    report = await advise(
        Picky,
        CASES,
        [code_check(lambda case, run: run.output != "tiny tiny")],
        candidates={"big": scripted("big"), "tiny": scripted("tiny")},
        baseline="big",
        prices=TABLE,
    )
    assert report.advised == {"stage:check": "tiny", "stage:draft": "tiny"}
    assert report.combined is not None and not report.combined.keeps_score
    assert report.mapping == {} and not report.verified
    assert report.note == "not verified: the advice together did not keep the score"
    assert "Advised one step at a time" in report.to_markdown()


def test_the_note_says_why_advice_was_not_verified():
    from agent_adapter.evals.advisor import _note

    assert (
        _note({"s": "m"}, None, True)
        == "not verified: the spend cap stopped before the combined run"
    )
    assert _note({}, None, True) == ""


class PaidLookups:
    """A target that pays for a lookup per case; the cheap model needs ten of them."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.name = "paid_lookups"

    async def run_case(self, case: EvalCase, *, executor: Executor, ctx: Any) -> str:
        reply = await executor.model(self.model).bind(ctx).generate(_ask("answer"))
        lookups = 10 if reply.model == "tiny" else 1
        for n in range(lookups):
            entry = LedgerEntry(run_id=ctx.run_id, call_id=f"lookup{n}", kind="api", cost_usd=0.01)
            ctx.ledger.record(entry)
        return reply.text


async def test_the_run_cost_counts_paid_api_calls_too():
    report = await advise(
        PaidLookups,
        CASES,
        [contains()],
        candidates={"big": scripted("big"), "tiny": scripted("tiny")},
        baseline="big",
        prices=TABLE,
    )
    base, (trial,) = report.baseline_trial, report.trials
    assert base is not None
    assert base.cost_usd == pytest.approx(base.model_cost_usd + 2 * 0.01)
    assert trial.model_cost_usd < base.model_cost_usd and trial.cost_usd > base.cost_usd
    assert report.mapping == {}  # cheaper model, dearer run: not advised


def test_the_report_round_trips_and_renders(tmp_path: Path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(c.model_dump_json() + "\n" for c in CASES), encoding="utf-8")
    report = advise_sync(
        loop_target,
        str(cases),
        [contains()],
        candidates={"big": scripted("big"), "tiny": scripted("tiny")},
        baseline="big",
        prices=TABLE,
        max_spend=1.0,
        name="pings",
    )
    again = AdvisorReport.from_json(report.to_json())
    assert again.mapping == report.mapping == {"stage:tool_loop": "tiny"}
    text = report.to_markdown()
    assert "# Cost advice: pings" in text and "| `stage:tool_loop` | 2 |" in text
    assert "Spent $" in text and "keeps the score" in text


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"candidates": {}}, "non-empty mapping"),
        ({"candidates": {"": scripted("big")}}, "non-empty strings"),
        ({"candidates": {"big": object()}}, "not an LLMPort"),
        ({"baseline": "nope"}, "baseline must name"),
        ({"min_score_ratio": 0}, "min_score_ratio"),
        ({"min_score_ratio": True}, "min_score_ratio"),
        ({"max_spend": -1}, "max_spend"),
        ({"max_spend": float("inf")}, "max_spend"),
        ({"prices": "table.json"}, "prices must be"),
        ({"max_spend": 1.0, "prices": Prices.empty("t")}, "needs a token price"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"max_output_tokens": True}, "max_output_tokens"),
    ],
)
async def test_bad_options_raise_config_error(options: dict[str, Any], match: str):
    given: dict[str, Any] = {"candidates": {"big": scripted("big")}, "baseline": "big"}
    given.update(options)
    with pytest.raises(ConfigError, match=match):
        await advise(loop_target, CASES, [contains()], **given)


async def test_the_target_factory_must_be_callable():
    with pytest.raises(ConfigError, match="target_factory"):
        await advise(
            "nope",  # type: ignore[arg-type]
            CASES,
            [contains()],
            candidates={"big": scripted("big")},
            baseline="big",
        )
