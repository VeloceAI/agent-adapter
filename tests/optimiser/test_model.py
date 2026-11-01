import pytest
from pydantic import ValidationError

from agent_adapter.optimiser.model import (
    Candidate,
    Edit,
    EvalCase,
    HarnessTree,
    OptimiserBudget,
    Score,
)


def test_with_edits_sets_deletes_and_inherits():
    base = HarnessTree(files={"prompt/system": "be brief", "tool/a/description": "A"})
    out = base.with_edits({"prompt/system": "be exact", "tool/a/description": None, "x/y": "z"})
    assert out.files == {"prompt/system": "be exact", "x/y": "z"}
    assert base.files == {"prompt/system": "be brief", "tool/a/description": "A"}


def test_with_edits_delete_of_missing_path_is_a_no_op():
    base = HarnessTree(files={"prompt/system": "s"})
    assert base.with_edits({"tool/a/prompt_hint": None}) == base


def test_diff_lists_added_removed_modified_and_char_delta():
    a = HarnessTree(files={"p/1": "aa", "p/2": "bb", "p/3": "cc"})
    b = HarnessTree(files={"p/1": "aa", "p/2": "bbbb", "p/4": "d"})
    d = a.diff(b)
    assert (d.added, d.removed, d.modified) == (["p/4"], ["p/3"], ["p/2"])
    assert d.char_delta == (2 + 4 + 1) - (2 + 2 + 2)


def test_unified_diff_headers_and_hunks():
    a = HarnessTree(files={"prompt/system": "one\ntwo"})
    b = HarnessTree(files={"prompt/system": "one\nthree"})
    text = a.unified_diff(b, a="a", b="b")
    assert text.startswith("--- a/prompt/system\n+++ b/prompt/system\n")
    assert "-two\n" in text and "+three\n" in text


def test_unified_diff_is_empty_for_equal_trees():
    a = HarnessTree(files={"prompt/system": "x"})
    assert a.unified_diff(a) == ""


def test_unified_diff_pretty_prints_json_configs():
    a = HarnessTree(files={"hook/budget/config.json": '{"max_usd": 1, "scope": "run"}'})
    b = HarnessTree(files={"hook/budget/config.json": '{"scope":"run","max_usd":2}'})
    text = a.unified_diff(b)
    assert '-  "max_usd": 1,\n' in text
    assert '+  "max_usd": 2,\n' in text
    assert '"scope": "run"' not in [line[1:] for line in text.splitlines() if line[:1] in "+-"]


def test_unified_diff_keeps_invalid_json_verbatim_and_handles_added_and_removed():
    a = HarnessTree(files={"hook/h/config.json": "{not json", "prompt/system": "s"})
    b = HarnessTree(files={"tool/t/prompt_hint": "hint"})
    text = a.unified_diff(b)
    assert "-{not json\n" in text
    assert "+++ candidate/tool/t/prompt_hint" in text
    assert "--- base/prompt/system" in text


def test_models_are_frozen_and_score_is_bounded():
    case = EvalCase(id="c1", input="q")
    with pytest.raises(ValidationError):
        case.id = "c2"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Score(value=1.5)
    assert Score(value=0.0).passed is None


def test_candidate_round_trips_through_json():
    cand = Candidate(
        label="proc1",
        dim="procedure",
        edits=(Edit(path="tool/a/description", content=None, reason="r"),),
        tree=HarnessTree(files={}),
        parent_label="base",
    )
    assert Candidate.model_validate_json(cand.model_dump_json()) == cand


def test_budget_defaults_are_unlimited():
    assert OptimiserBudget() == OptimiserBudget(max_usd=None, max_case_runs=None, max_wall_s=None)
