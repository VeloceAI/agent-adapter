"""Render an `EvalReport` as JUnit XML or Markdown."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_adapter.evals.report import CaseResult, EvalReport

_REASON_CHARS = 2_000  # per failure message
_NOT_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")  # not in XML 1.0


def _x(text: str) -> str:
    """`text` without the characters XML 1.0 cannot hold."""
    return _NOT_XML.sub("\N{REPLACEMENT CHARACTER}", text)


def _reason(case: CaseResult) -> str:
    """Why a case did not pass: its error, else the notes of the grades that failed."""
    if case.error:
        return case.error
    notes = [
        f"{g.grader}: {g.notes or ('not graded' if g.score is None else 'failed')}"
        for g in case.grades
        if not g.passed
    ]
    return "; ".join(notes)[:_REASON_CHARS] or "failed"


def junit(report: EvalReport) -> str:
    """JUnit XML (Ant/Surefire shape): one testcase per case; run id and score in system-out."""
    failures = sum(1 for c in report.cases if not c.passed and not c.error)
    suites = ET.Element("testsuites", name=_x(report.name))
    suite = ET.SubElement(
        suites,
        "testsuite",
        name=_x(report.name),
        tests=str(report.n_cases),
        failures=str(failures),
        errors=str(report.n_errors),
        skipped="0",
        time=f"{report.wall_s:.3f}",
        timestamp=report.started_at[:19],  # the schema wants no UTC offset
    )
    for case in report.cases:
        element = ET.SubElement(
            suite,
            "testcase",
            classname=_x(f"evals.{report.name}"),
            name=_x(case.case_id),
            time=f"{case.latency_s:.3f}",
        )
        score = "" if case.score is None else f"{case.score:.4f}"
        if case.error:
            error = _x(case.error[:_REASON_CHARS])
            ET.SubElement(element, "error", message=error, type=error.split(":")[0])
        elif not case.passed:
            reason = _x(_reason(case))
            ET.SubElement(element, "failure", message=reason, type="eval-failed").text = reason
        facts = f"run_id: {case.run_id}\nscore: {score}\ntags: {','.join(case.tags)}"
        ET.SubElement(element, "system-out").text = _x(facts)
    ET.indent(suites)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(suites, encoding="unicode")


def _cell(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return " ".join(text.replace("|", "\\|").split())


def markdown(report: EvalReport) -> str:
    """Totals, a per-tag table, a per-grader table and the cases that did not pass."""
    lines = [
        f"# Eval: {_cell(report.name)}",
        "",
        f"- Cases: {report.n_cases}, passed {report.n_passed}, errors {report.n_errors}",
        f"- Mean score: {report.mean_score:.3f}, pass rate {report.pass_rate:.1%}",
        f"- Cost: ${report.cost_usd:.4f}, wall time {report.wall_s:.2f}s",
        "",
    ]
    for title, groups in (("Tag", report.by_tag), ("Grader", report.by_grader)):
        if groups:
            lines += [f"| {title} | Cases | Passed | Mean score |", "|---|---|---|---|"]
            lines += [
                f"| {_cell(name)} | {g.n} | {g.n_passed} | {g.mean_score:.3f} |"
                for name, g in groups.items()
            ]
            lines.append("")
    failed = report.failed
    if failed:
        lines += ["## Not passed", "", "| Case | Score | Run | Reason |", "|---|---|---|---|"]
        for case in failed:
            score = "-" if case.score is None else f"{case.score:.3f}"
            lines.append(
                f"| {_cell(case.case_id)} | {score} | `{case.run_id}` | {_cell(_reason(case))} |"
            )
        lines.append("")
    return "\n".join(lines)
