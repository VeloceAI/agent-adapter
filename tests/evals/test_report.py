import json
import xml.etree.ElementTree as ET

import pytest

from agent_adapter.evals import CaseResult, EvalReport, GradeResult


def result(case_id, passed=True, score=1.0, *, tags=(), error="", notes="", cost=0.0):
    grade = GradeResult(grader="exact", score=score, passed=passed, notes=notes)
    return CaseResult(
        case_id=case_id,
        tags=tags,
        run_id=f"run_{case_id}",
        passed=passed,
        score=score,
        grades=() if error else (grade,),
        error=error,
        cost_usd=cost,
        latency_s=0.5,
    )


def report(*cases, name="shop"):
    return EvalReport(name=name, started_at="2026-01-01T00:00:00+00:00", wall_s=1.0, cases=cases)


def test_totals_count_errors_and_ungraded_cases_as_zero():
    made = report(
        result("a", tags=("orders",), cost=0.1),
        result("b", False, 0.5, tags=("orders", "chat")),
        result("c", False, None, error="timeout: slow"),
        result("d", False, None),
    )
    assert (made.n_cases, made.n_passed, made.n_errors) == (4, 1, 1)
    assert made.mean_score == pytest.approx(1.5 / 4)
    assert made.pass_rate == 0.25 and made.cost_usd == pytest.approx(0.1)
    assert made.by_tag["orders"].n == 2 and made.by_tag["orders"].mean_score == 0.75
    assert made.by_tag["chat"].n_passed == 0
    assert made.by_grader["exact"].n == 3 and made.by_grader["exact"].n_passed == 1
    assert [c.case_id for c in made.failed] == ["b", "c", "d"]


def test_an_empty_report_has_zero_totals():
    empty = report()
    assert empty.mean_score == 0.0 and empty.pass_rate == 0.0 and empty.by_tag == {}


def test_json_round_trips_with_totals():
    made = report(result("a", tags=("x",)), result("b", False, 0.0, notes="wrong"))
    data = json.loads(made.to_json())
    assert data["n_passed"] == 1 and data["by_tag"]["x"]["n"] == 1
    assert data["cases"][1]["run_id"] == "run_b"
    assert EvalReport.from_json(made.to_json()) == made


def test_compare_finds_regressions_improvements_and_changed_cases():
    baseline = report(result("a"), result("b", False, 0.0), result("c"), result("gone"))
    now = report(result("a", False, 0.0), result("b"), result("c", True, 0.9), result("new"))
    diff = now.compare(baseline)
    assert [c.case_id for c in diff.regressions] == ["a", "c"]
    assert [c.case_id for c in diff.improvements] == ["b"]
    assert diff.added == ("new",) and diff.removed == ("gone",) and diff.regressed
    only_removed = report(result("a")).compare(report(result("a"), result("gone")))
    assert only_removed.regressions == () and only_removed.regressed
    assert diff.score_delta == pytest.approx(now.mean_score - baseline.mean_score)
    lenient = now.compare(baseline, tolerance=0.2)
    assert [c.case_id for c in lenient.regressions] == ["a"]
    assert not report(result("a")).compare(report(result("a"))).regressed


def test_junit_marks_failures_and_errors_and_strips_bad_xml_characters():
    made = report(
        result("ok"),
        result("bad\x00id", False, 0.0, notes="missing <answer> & more"),
        result("err", False, None, error="timeout: slow"),
        name="shop",
    )
    root = ET.fromstring(made.to_junit().split("\n", 1)[1])
    suite = root.find("testsuite")
    assert suite.attrib["tests"] == "3"
    assert suite.attrib["failures"] == "1" and suite.attrib["errors"] == "1"
    cases = suite.findall("testcase")
    assert cases[1].attrib["name"] == "bad�id"
    assert "missing <answer> & more" in cases[1].find("failure").attrib["message"]
    assert cases[2].find("error").attrib["type"] == "timeout"
    assert cases[0].find("properties") is None  # not allowed in a testcase by the XSD
    assert cases[0].find("system-out").text == "run_id: run_ok\nscore: 1.0000\ntags: "
    assert suite.attrib["timestamp"] == "2026-01-01T00:00:00"


def test_markdown_lists_totals_tags_graders_and_failures():
    made = report(
        result("a", tags=("orders",)),
        result("b|c", False, 0.25, notes="wrong\nanswer"),
        result("e", False, None, error="timeout: slow"),
    )
    text = made.to_markdown()
    assert "# Eval: shop" in text and "Cases: 3, passed 1, errors 1" in text
    assert "| orders | 1 | 1 | 1.000 |" in text and "| exact |" in text
    assert "| b\\|c | 0.250 | `run_b|c` | exact: wrong answer |" in text
    assert "| e | - | `run_e` | timeout: slow |" in text


def test_markdown_without_failures_or_tags_is_short():
    text = report(result("a")).to_markdown()
    assert "Not passed" not in text and "| Tag |" not in text


def test_a_failure_without_notes_still_says_why():
    ungraded = CaseResult(
        case_id="u",
        passed=False,
        score=None,
        grades=(GradeResult(grader="g", score=None, passed=False),),
    )
    assert "g: not graded" in report(ungraded).to_markdown()
    no_grades = CaseResult(case_id="n", passed=False, score=None)
    assert "| n | - | `` | failed |" in report(no_grades).to_markdown()


def test_junit_clips_long_errors_and_drops_lone_surrogates():
    made = report(result("e", False, None, error="tool-failed: " + "x" * 5000 + "\ud800"))
    text = made.to_junit()
    text.encode("utf-8")
    message = ET.fromstring(text.split("\n", 1)[1]).find("testsuite/testcase/error")
    assert len(message.attrib["message"]) == 2000
