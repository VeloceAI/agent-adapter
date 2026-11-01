import pytest

from agent_adapter.core.errors import (
    BudgetExceeded,
    ModelAuthError,
    ModelUnavailable,
    OptimiserError,
)
from agent_adapter.optimiser._select import select_final
from agent_adapter.optimiser.model import Candidate, CaseRun, EvalCase, HarnessTree
from agent_adapter.optimiser.probe import summarise
from tests.optimiser.fakes import ScriptedProposer

BASE_TREE = HarnessTree(files={"prompt/system": "base"})
CASES = {"q0": EvalCase(id="q0", input="first"), "q1": EvalCase(id="q1", input="second")}


def report(label, scores, *, delivered=(True, True), outputs=("a", "b")):
    return summarise(
        label,
        [
            CaseRun(
                case_id=f"q{i}", label=label, ok=True, delivered=d, output=o, transcript="", score=s
            )
            for i, (s, d, o) in enumerate(zip(scores, delivered, outputs, strict=True))
        ],
    )


def final(label, system):
    tree = BASE_TREE.with_edits({"prompt/system": system})
    return Candidate(label=label, dim="compose", edits=(), tree=tree, parent_label="base")


async def pick(base, finals, **kwargs):
    options = {"min_gain": 0.0, "seed": 0, "cases": CASES, "selector": None, "judged": True}
    options.update(kwargs)
    return await select_final(base, BASE_TREE, finals, **options)


async def test_judge_ranked_selection_picks_the_best_final_that_beats_base():
    base = report("base", [0.0, 0.5])
    finals = [
        (final("agg1", "longer system prompt"), report("agg1", [1.0, 0.5])),
        (final("agg2", "short"), report("agg2", [1.0, 0.5])),
        (final("agg3", "x"), report("agg3", [1.0, 1.0], delivered=(True, False))),
    ]
    choice = await pick(base, finals)
    assert (choice.chosen, choice.calls) == ("agg2", 0)  # tie on score -> fewer always-on chars
    assert "agg3" in choice.why and "fewer answers" in choice.why


async def test_no_final_beats_base_keeps_base():
    base = report("base", [1.0, 0.5])
    choice = await pick(base, [(final("agg1", "y"), report("agg1", [1.0, 0.5]))])
    assert choice.chosen is None and "KEEP_BASE" in choice.why
    same = await pick(
        report("base", [0.0, 0.0]),
        [
            (
                Candidate(
                    label="agg1", dim="compose", edits=(), tree=BASE_TREE, parent_label="base"
                ),
                report("agg1", [1.0, 1.0]),
            )
        ],
    )
    assert same.chosen is None


async def test_min_gain_must_be_met():
    base = report("base", [0.0, 0.5])
    finals = [(final("agg1", "y"), report("agg1", [0.2, 0.5]))]
    assert (await pick(base, finals, min_gain=0.2)).chosen is None
    assert (await pick(base, finals, min_gain=0.05)).chosen == "agg1"


async def test_blind_selection_without_a_judge():
    base = report("base", [None, None], outputs=("base-a", "base-b"))
    finals = [(final("agg1", "y"), report("agg1", [None, None], outputs=("new-a", "new-b")))]
    selector = ScriptedProposer({"select": {"why": "answers the question", "choice": "X"}})
    first = await pick(base, finals, judged=False, selector=selector.port)
    assert first.calls == 1 and selector.stages() == ["select"]
    text = selector.requests[0].messages[1].content
    letters = [line.split()[-1] for line in text.splitlines() if line.startswith("# Answer set")]
    assert letters == ["A", "B"]
    assert "new-a" in text and "base-b" in text and "prompt/system" not in text
    assert first.chosen is None and "selector failed" in first.why  # "X" is not a letter
    agg_letter = "A" if text.index("new-a") < text.index("base-a") else "B"
    selector = ScriptedProposer({"select": {"why": "better", "choice": agg_letter}})
    chosen = await pick(base, finals, judged=False, selector=selector.port)
    assert chosen.chosen == "agg1" and chosen.why == "better"
    base_letter = "B" if agg_letter == "A" else "A"
    selector = ScriptedProposer({"select": {"why": "base wins", "choice": base_letter}})
    kept = await pick(base, finals, judged=False, selector=selector.port)
    assert kept.chosen is None and "base wins" in kept.why


async def test_blind_selection_skips_the_call_when_nothing_differs_and_survives_errors():
    base = report("base", [None, None])
    same = [
        (
            Candidate(label="agg1", dim="compose", edits=(), tree=BASE_TREE, parent_label="base"),
            report("agg1", [None, None]),
        )
    ]
    selector = ScriptedProposer({})
    assert (await pick(base, same, judged=False, selector=selector.port)).calls == 0
    finals = [(final("agg1", "y"), report("agg1", [None, None]))]
    broken = ScriptedProposer({"select": ModelUnavailable("down", provider="openai")})
    out = await pick(base, finals, judged=False, selector=broken.port)
    assert out.chosen is None and "model-unavailable" in out.why and out.calls == 1
    stop = ScriptedProposer({"select": BudgetExceeded(1.0, 1.0)})
    with pytest.raises(BudgetExceeded):
        await pick(base, finals, judged=False, selector=stop.port)


async def test_blind_selection_prompt_stays_within_the_evidence_budget():
    cases = {f"q{i}": EvalCase(id=f"q{i}", input="x" * 500) for i in range(60)}

    def big(label, letter):
        runs = [
            CaseRun(
                case_id=f"q{i}",
                label=label,
                ok=True,
                delivered=True,
                output=letter * 3000,
                transcript="",
            )
            for i in range(60)
        ]
        return summarise(label, runs)

    selector = ScriptedProposer({"select": {"why": "w", "choice": "A"}})
    await select_final(
        big("base", "a"),
        BASE_TREE,
        [(final("agg1", "y"), big("agg1", "b"))],
        judged=False,
        min_gain=0.0,
        seed=0,
        cases=cases,
        selector=selector.port,
        budget=60_000,
    )
    text = selector.requests[0].messages[1].content
    assert len(text) <= 60_000
    assert "# Answer set A" in text and "# Answer set B" in text
    assert text.count("## Case q0") == 2  # the same cases are shown for every set


async def test_fatal_selector_errors_raise():
    base = report("base", [None, None])
    finals = [(final("agg1", "y"), report("agg1", [None, None]))]
    denied = ScriptedProposer({"select": ModelAuthError("bad key", provider="openai")})
    with pytest.raises(OptimiserError, match="model-auth-failed"):
        await pick(base, finals, judged=False, selector=denied.port)
