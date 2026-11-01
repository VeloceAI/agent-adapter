import json

from agent_adapter.cost.ledger import Ledger
from agent_adapter.optimiser.model import (
    BudgetUsage,
    Candidate,
    CaseRun,
    DecisionLine,
    HarnessTree,
    OptimiserBudget,
    OptimiserConfig,
    OptimiseResult,
)
from agent_adapter.optimiser.probe import summarise
from agent_adapter.optimiser.report import (
    case_file,
    case_stems,
    patch_for,
    render_report,
    result_json,
    write_outputs,
)

BASE = HarnessTree(files={"prompt/system": "a", "hook/h/config.json": '{"x": 1}'})
NEW = BASE.with_edits({"prompt/system": "b"})


def test_case_file_names_are_safe_and_unique():
    stems = case_stems(["q1", "a/b:c", "a_b:c", "CON", "nul.txt", ""])
    assert case_file("base", stems["q1"]) == "runs/base/q1.md"
    assert stems["a/b:c"].startswith("a_b_c-")
    assert stems["a/b:c"] != stems["a_b:c"]
    assert stems["CON"].startswith("CON-") and stems["nul.txt"].startswith("nul.txt-")
    assert stems[""].startswith("case-")


def test_patch_for_keep_base_is_empty():
    assert patch_for(BASE, None) == ""
    assert patch_for(BASE, NEW).startswith("--- a/prompt/system")


def result(**overrides):
    run = CaseRun(
        case_id="q|1",
        label="base",
        ok=False,
        delivered=False,
        output="o",
        transcript="[USER] secret transcript",
        error="x" * 300,
    )
    base = summarise("base", [run])
    cand = Candidate(label="base", dim="base", edits=(), tree=BASE, parent_label="")
    fell = Candidate(
        label="know1",
        dim="knowledge",
        edits=(),
        tree=BASE,
        parent_label="base",
        fell_back=True,
        error="too long",
    )
    data = {
        "base": base,
        "candidates": {"base": cand, "know1": fell},
        "probes": {"base": base},
        "finals": [],
        "chosen": None,
        "decisions": [DecisionLine(kind="dropped", path="prompt/system", evidence="[agg1] x")],
        "patch": "",
        "report_md": "",
        "lessons_added": [],
        "ledger_summary": Ledger().summary(),
        "budget": BudgetUsage(
            usd=0.5, case_runs=1, wall_s=2.0, proposer_calls=0, merge_calls=0, stopped="max_usd"
        ),
        "errors": ["stopped early: max_usd"],
    }
    data.update(overrides)
    return OptimiseResult(**data)


def test_report_marks_partial_runs_and_escapes_cells():
    config = OptimiserConfig(budget=OptimiserBudget(max_usd=0.5))
    text = render_report(
        result(),
        run_id="opt_1",
        domain="tax",
        config=config,
        n_cases=1,
        why="No final was selected.",
    )
    assert "**Stopped early: max_usd.**" in text
    assert "| case runs | 1 | (1 + 3k) x N = 4 |" in text
    assert "| cost (USD) | 0.5000 | limit 0.5 |" in text
    assert "| q\\|1 | no |" in text and "..." in text
    assert "fell back: too long" in text
    assert "- dropped -> `prompt/system` -> [agg1] x" in text
    assert "Chosen: **KEEP_BASE**" in text


def test_result_json_references_transcripts_by_path(tmp_path):
    data = json.loads(result_json(result()))
    assert data["base"]["runs"][0]["transcript"] == case_file("base", case_stems(["q|1"])["q|1"])
    assert "secret transcript" not in result_json(result())
    write_outputs(tmp_path, result(), run_id="opt_1", chosen_tree=BASE, cases={})
    case_text = (tmp_path / case_file("base", case_stems(["q|1"])["q|1"])).read_text("utf-8")
    assert "secret transcript" in case_text and "error: " in case_text
    after = json.loads((tmp_path / "harness.after.json").read_text("utf-8"))
    assert after == {"version": 1, "run": "opt_1", "label": "base", "files": BASE.files}
    assert (tmp_path / "harness.patch").read_text("utf-8") == ""


def test_case_files_do_not_collide_on_case_insensitive_file_systems():
    stems = case_stems(["Q1", "q1", "x", "CON"])
    assert len({s.lower() for s in stems.values()}) == 4
    assert stems["x"] == "x"
    assert stems["Q1"].startswith("Q1-") and stems["q1"].startswith("q1-")
    assert case_file("base", stems["Q1"]) == f"runs/base/{stems['Q1']}.md"


def test_report_lists_reflection_and_selection_calls():
    text = render_report(
        result(), run_id="opt_1", domain="tax", config=OptimiserConfig(), n_cases=1, why="w"
    )
    assert "| reflection calls | 0 | at most 1 |" in text
    assert "| selection calls | 0 | at most 1, only without a judge |" in text


async def test_outputs_keep_two_case_ids_differing_only_in_case_apart(tmp_path):
    runs = [
        CaseRun(case_id=cid, label="base", ok=True, delivered=True, output=cid, transcript=cid)
        for cid in ("Q1", "q1")
    ]
    base = summarise("base", runs)
    write_outputs(
        tmp_path,
        result(base=base, probes={"base": base}),
        run_id="opt_1",
        chosen_tree=BASE,
        cases={},
    )
    files = sorted(p.name for p in (tmp_path / "runs" / "base").iterdir())
    assert len(files) == 2
