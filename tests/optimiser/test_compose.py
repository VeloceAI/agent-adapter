import json

import pytest

from agent_adapter.core.errors import BudgetExceeded, ModelBadRequest, OptimiserError
from agent_adapter.optimiser.compose import assess, compose
from agent_adapter.optimiser.model import Candidate, CaseRun, Edit, HarnessTree
from agent_adapter.optimiser.probe import summarise
from tests.optimiser.fakes import FakeHost, ScriptedProposer

HOST = FakeHost()
BASE_TREE = HOST.surface()
BASE = Candidate(label="base", dim="base", edits=(), tree=BASE_TREE, parent_label="")


def runs(label, scores, *, delivered=None, metrics=None):
    delivered = delivered or [True] * len(scores)
    return summarise(
        label,
        [
            CaseRun(
                case_id=f"q{i}",
                label=label,
                ok=True,
                delivered=d,
                output="o",
                transcript="t",
                score=s,
                passed=None if s is None else s >= 0.5,
                metrics=metrics or {},
            )
            for i, (s, d) in enumerate(zip(scores, delivered, strict=True))
        ],
    )


REF = runs("base", [0.0, 0.0, 0.5])


def compose_pool(pool, *, mode="greedy", host=HOST, label="agg1", **options):
    """`compose` of `pool` onto BASE against REF, with no minimum gain and seed 0."""
    return compose(
        label,
        BASE,
        REF,
        pool,
        caps=host.caps(),
        host=host,
        mode=mode,
        min_gain=0.0,
        seed=0,
        **options,
    )


def cand(label, *edits, dim="procedure", fell_back=False):
    made = tuple(Edit(path=p, content=c, reason="r") for p, c in edits)
    return Candidate(
        label=label,
        dim=dim,
        edits=made,
        tree=BASE_TREE.with_edits(dict(edits)),
        parent_label="base",
        fell_back=fell_back,
        error="bad" if fell_back else "",
    )


def two_way_conflict():
    """know1 and proc1 both edit prompt/system; proc1 gains more."""
    a = cand("know1", ("prompt/system", "cite A"), dim="knowledge")
    b = cand("proc1", ("prompt/system", "cite B"))
    return [(a, runs("know1", [1.0, 0.0, 0.5])), (b, runs("proc1", [1.0, 1.0, 0.5]))]


def test_assess_admits_a_safe_improving_exercised_candidate():
    a = assess(cand("p1", ("prompt/system", "x")), runs("p1", [1.0, 0.0, 0.5]), REF, min_gain=0.0)
    assert a.admitted and a.gain == pytest.approx(1 / 3)
    assert (a.improved, a.regressed, a.undelivered) == (1, 0, 0)
    assert a.evidence() == "improved 1/3, regressed 0/3, undelivered 0, gain +0.333"


@pytest.mark.parametrize(
    ("candidate", "report", "reason"),
    [
        (cand("p", ("prompt/system", "x"), fell_back=True), runs("p", [1, 1, 1]), "fell back: bad"),
        (cand("p", ("prompt/system", "x")), None, "not probed"),
        (
            cand("p", ("prompt/system", "x")),
            runs("p", [1.0, 1.0, 0.5], delivered=[True, True, False]),
            "unsafe: 1 case(s) no longer answered",
        ),
        (cand("p", ("prompt/system", "x")), runs("p", [0.0, 0.0, 0.5]), "no case improved"),
        (cand("p", ("prompt/system", "x")), runs("p", [1.0, 0.0, 0.0]), "gain +0.167 below 0.2"),
        (
            cand("p", ("tool/search/prompt_hint", "x")),
            runs("p", [1.0, 1.0, 1.0], metrics={"tool_calls": 0}),
            "never exercised",
        ),
    ],
)
def test_assess_drops_candidates(candidate, report, reason):
    ref = runs("base", [0.0, 0.0, 0.5], metrics={"tool_calls": 0})
    a = assess(candidate, report, ref, min_gain=0.2 if "below" in reason else 0.0)
    assert not a.admitted and a.reason == reason


def test_assess_without_scores_uses_delivery():
    ref = runs("base", [None, None], delivered=[True, False])
    better = assess(cand("p", ("prompt/system", "x")), runs("p", [None, None]), ref, min_gain=0.0)
    assert better.admitted and better.gain is None and better.improved == 1
    assert better.evidence() == "improved 1/2, regressed 0/2, undelivered 0, gain n/a"


async def test_greedy_compose_merges_disjoint_edits_and_records_decisions():
    p1 = cand("proc1", ("tool/search/prompt_hint", "search first"))
    k1 = cand("know1", ("prompt/system", "cite the section"), dim="knowledge")
    k2 = cand("know2", ("prompt/system", "EMPTY"), dim="knowledge")
    pool = [
        (p1, runs("proc1", [1.0, 0.0, 0.5])),
        (k1, runs("know1", [1.0, 1.0, 0.5])),
        (k2, runs("know2", [1.0, 1.0, 1.0], delivered=[False, True, True])),
    ]
    out = await compose_pool(pool)
    assert out.merge_calls == 0
    assert out.candidate.dim == "compose" and out.candidate.parent_label == "base"
    assert out.candidate.tree.files["prompt/system"] == "cite the section"
    assert out.candidate.tree.files["tool/search/prompt_hint"] == "search first"
    lines = [(d.kind, d.path, d.evidence.split(":")[0]) for d in out.decisions]
    assert ("kept", "prompt/system", "know1") in lines
    assert ("kept", "tool/search/prompt_hint", "proc1") in lines
    assert ("dropped", "prompt/system", "know2") in lines


async def test_greedy_conflict_takes_the_highest_gain():
    pool = two_way_conflict()
    out = await compose_pool(pool)
    assert out.candidate.tree.files["prompt/system"] == "cite B"
    revised = [d for d in out.decisions if d.kind == "revised"]
    assert revised[0].path == "prompt/system"
    assert revised[0].evidence.startswith("conflict: took proc1 over know1")


async def test_llm_compose_merges_conflicts_with_one_call():
    pool = two_way_conflict()
    merger = ScriptedProposer(
        {
            "agg1": {
                "merges": [{"path": "prompt/system", "content": "cite A and B", "decision": "both"}]
            }
        }
    )
    out = await compose_pool(pool, mode="llm", composer=merger.port)
    assert out.merge_calls == 1 and merger.stages() == ["merge:agg1"]
    assert out.candidate.tree.files["prompt/system"] == "cite A and B"
    user = merger.requests[0].messages[1].content
    assert "cite A" in user and "cite B" in user and "improved" in user
    assert [d.evidence for d in out.decisions if d.kind == "revised"] == ["merged by LLM: both"]


@pytest.mark.parametrize(
    "reply",
    [
        {"merges": [{"path": "tool/other/description", "content": "x", "decision": "d"}]},
        {"merges": [{"path": "prompt/system", "content": "x" * 20_000, "decision": "d"}]},
        "not json",
        {"merges": []},
    ],
)
async def test_llm_merge_failure_falls_back_to_greedy(reply):
    pool = two_way_conflict()
    merger = ScriptedProposer({"agg1": reply})
    out = await compose_pool(pool, mode="llm", composer=merger.port)
    assert out.merge_calls == 1
    assert out.candidate.tree.files["prompt/system"] == "cite B"
    assert any("llm merge failed" in d.evidence for d in out.decisions)


async def test_budget_stop_during_merge_propagates():
    pool = two_way_conflict()
    merger = ScriptedProposer({"agg1": BudgetExceeded(1.0, 2.0)})
    with pytest.raises(BudgetExceeded):
        await compose_pool(pool, mode="llm", composer=merger.port)


async def test_nothing_admitted_keeps_the_parent_and_keep_limits_the_mix():
    out = await compose_pool([])
    assert out.candidate.edits == () and out.candidate.tree == BASE_TREE
    p1 = cand("proc1", ("tool/search/prompt_hint", "search first"))
    k1 = cand("know1", ("prompt/system", "cite"), dim="knowledge")
    pool = [(p1, runs("proc1", [1.0, 0.0, 0.5])), (k1, runs("know1", [1.0, 1.0, 0.5]))]
    two = await compose_pool(pool, label="agg2", drop=1)
    assert two.candidate.tree.files["prompt/system"] == "cite"
    assert "tool/search/prompt_hint" not in two.candidate.tree.files
    assert any(d.evidence.startswith("proc1: left out of agg2") for d in two.decisions)


async def test_invalid_combination_falls_back_to_the_best_candidate():
    def no_both(tree: HarnessTree):
        both = "tool/search/prompt_hint" in tree.files and "cite" in tree.files["prompt/system"]
        return ["hint and citation conflict"] if both else []

    host = FakeHost(problems=no_both)
    p1 = cand("proc1", ("tool/search/prompt_hint", "search first"))
    k1 = cand("know1", ("prompt/system", "cite"), dim="knowledge")
    pool = [(p1, runs("proc1", [1.0, 0.0, 0.5])), (k1, runs("know1", [1.0, 1.0, 0.5]))]
    out = await compose_pool(pool, host=host)
    assert out.candidate.tree == k1.tree
    assert any("hint and citation conflict" in d.evidence for d in out.decisions)
    assert json.loads(out.candidate.tree.files["hook/budget/config.json"]) == {"max_usd": 5.0}


def conflict_pool():
    a = cand("know1", ("prompt/system", "cite A"), dim="knowledge")
    b = cand("proc1", ("prompt/system", "cite B"))
    c = cand("proc2", ("tool/search/prompt_hint", "search first"))
    return [
        (a, runs("know1", [1.0, 0.0, 0.5])),
        (b, runs("proc1", [1.0, 1.0, 0.5])),
        (c, runs("proc2", [0.5, 0.0, 0.5])),
    ]


async def test_an_invalid_llm_merge_falls_back_to_the_greedy_tree_first():
    def no_merged(tree: HarnessTree):
        return ["merged text rejected"] if "A+B" in tree.files["prompt/system"] else []

    host = FakeHost(problems=no_merged)
    merger = ScriptedProposer(
        {"agg1": {"merges": [{"path": "prompt/system", "content": "cite A+B", "decision": "d"}]}}
    )
    out = await compose_pool(conflict_pool(), mode="llm", host=host, composer=merger.port)
    assert out.candidate.tree.files["prompt/system"] == "cite B"
    assert out.candidate.tree.files["tool/search/prompt_hint"] == "search first"
    assert any("merged harness invalid" in d.evidence for d in out.decisions)


async def test_fatal_merge_errors_raise():
    merger = ScriptedProposer({"agg1": ModelBadRequest("bad", provider="openai")})
    with pytest.raises(OptimiserError, match="model-bad-request"):
        await compose_pool(conflict_pool(), mode="llm", composer=merger.port)
