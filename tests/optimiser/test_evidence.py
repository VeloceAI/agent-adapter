from agent_adapter.optimiser.evidence import (
    build_evidence,
    clip,
    clip_observations,
    render_lessons,
    worst_first,
)
from agent_adapter.optimiser.model import CaseRun, EvalCase, HarnessTree, Lesson, ProbeReport


def run(case_id, *, delivered=True, score=None, transcript="", output="out", notes="", error=""):
    return CaseRun(
        case_id=case_id,
        label="base",
        ok=not error,
        delivered=delivered,
        output=output,
        transcript=transcript,
        score=score,
        passed=None if score is None else score >= 0.5,
        judge_notes=notes,
        error=error,
    )


def report(*runs):
    return ProbeReport(
        label="base",
        runs=list(runs),
        mean_score=None,
        n_passed=0,
        n_graded=0,
        n_delivered=sum(r.delivered for r in runs),
        cost_usd=0.0,
    )


CASES = {
    "a": EvalCase(id="a", input="question a", expected="SECRET-ANSWER", tags=("x",)),
    "b": EvalCase(id="b", input={"q": "b"}, expected="SECRET-ANSWER"),
    "c": EvalCase(id="c", input="question c"),
}
TREE = HarnessTree(files={"prompt/system": "Be exact.", "tool/s/description": "Search."})


def test_clip_keeps_short_text_and_marks_long_text():
    assert clip("short", 100) == "short"
    out = clip("x" * 500, 100)
    assert len(out) == 100 and out.endswith("[... 424 chars clipped]")
    assert clip("abcdef", 3) == "abc"


def test_clip_observations_clips_each_block_separately():
    transcript = "[USER] hi\n[OBSERVATION] " + "y" * 300 + "\n[ASSISTANT] done"
    out = clip_observations(transcript, 60)
    assert out.startswith("[USER] hi\n[OBSERVATION] yyy")
    assert "chars clipped]" in out
    assert out.endswith("\n[ASSISTANT] done")


def test_worst_first_puts_undelivered_then_low_scores_first():
    runs = [run("a", score=0.9), run("b", delivered=False), run("c", score=0.1)]
    assert [r.case_id for r in worst_first(runs)] == ["b", "c", "a"]


def test_evidence_hides_scores_and_answers_by_default():
    text = build_evidence(
        tree=TREE,
        cases=CASES,
        report=report(run("a", score=0.0, notes="misses the section"), run("b", score=1.0)),
        obs_trunc=1000,
        budget=10_000,
        show_scores=False,
    )
    assert "# CURRENT HARNESS" in text and "Be exact." in text
    assert "misses the section" in text
    assert "question a" in text and '{"q": "b"}' in text
    assert "SECRET-ANSWER" not in text
    assert "0.00" not in text and "Result:" not in text
    shown = build_evidence(
        tree=TREE,
        cases=CASES,
        report=report(run("a", score=0.0)),
        obs_trunc=1000,
        budget=10_000,
        show_scores=True,
    )
    assert "Result: 0.00 (not passed)" in shown


def test_evidence_is_bounded_and_keeps_every_case_visible():
    big = "[OBSERVATION] " + "z" * 50_000
    text = build_evidence(
        tree=TREE,
        cases=CASES,
        report=report(*(run(c, transcript=big, output="o" * 5000) for c in "abc")),
        obs_trunc=4_000,
        budget=6_000,
        show_scores=False,
        lessons_text="- lesson",
        history="h" * 10_000,
    )
    assert len(text) <= 6_000
    for case_id in "abc":
        assert f"## Case {case_id}" in text
    assert "# HISTORY" in text and "# EARLIER RUNS" in text


def test_render_lessons_lists_cues():
    lesson = Lesson(
        id="ls_1",
        domain="tax",
        created_at="2026-10-30T00:00:00+00:00",
        cue="when a search returns nothing",
        lesson="in tool/search/prompt_hint, broaden once then stop",
        evidence="proc1: 2/3 improved",
        confidence="medium",
        component_paths=["tool/search/prompt_hint"],
        run="runs/opt",
        adopted=True,
    )
    text = render_lessons([lesson])
    assert "when a search returns nothing" in text and "(medium)" in text
    assert render_lessons([]) == ""


def test_evidence_shows_only_the_weakest_cases_when_there_are_too_many():
    many = {f"c{i:03}": EvalCase(id=f"c{i:03}", input=f"q{i}") for i in range(100)}
    runs = [
        run(cid, score=i / 100, transcript="[OBSERVATION] " + "x" * 2000)
        for i, cid in enumerate(many)
    ]
    text = build_evidence(
        tree=TREE,
        cases=many,
        report=report(*runs),
        obs_trunc=4000,
        budget=10_000,
        show_scores=False,
    )
    assert len(text) <= 10_000
    assert "# CASES (100, weakest first; the 12 weakest shown)" in text
    assert "## Case c000" in text and "## Case c011" in text and "## Case c012" not in text
