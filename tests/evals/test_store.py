"""Saving eval reports for the viewer: `save_report`, `run_eval(save_to=)`, `eval --save`."""

import argparse
import io
import json
import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_adapter.cli.eval import add_eval_parser, run_eval_command
from agent_adapter.core.errors import ConfigError
from agent_adapter.evals import (
    DEFAULT_EVALS_FOLDER,
    EvalCase,
    EvalReport,
    run_eval_sync,
    save_report,
)
from agent_adapter.evals.graders import exact
from agent_adapter.evals.store import REPORT_NAME, report_key

CASES = [
    EvalCase(id="a", input="1042", expected="shipped"),
    EvalCase(id="b", input={"order": "1043"}),
]


def answer(question: object) -> str:
    return "shipped"


def test_a_report_is_saved_as_json_the_viewer_can_list(tmp_path: Path) -> None:
    report = run_eval_sync(answer, CASES, [exact()], name="shop support / v2")
    path = save_report(report, tmp_path / "evals")
    assert REPORT_NAME.match(path.name) and "-shop-support-v2-" in path.name
    assert EvalReport.from_json(path.read_text(encoding="utf-8")) == report
    assert [p.name for p in path.parent.iterdir()] == [path.name], "no partial file left"
    assert save_report(report, tmp_path / "evals") != path, "each save is a new file"


def test_cases_keep_their_input_and_expected_text() -> None:
    report = run_eval_sync(answer, CASES, [exact()])
    first, second = report.cases
    assert (first.input, first.expected) == ("1042", "shipped")
    assert (second.input, second.expected) == ('{"order": "1043"}', "")
    old = json.loads(report.to_json())
    for item in old["cases"]:
        del item["input"], item["expected"]
    assert EvalReport.from_json(json.dumps(old)).cases[0].input == "", "older reports still load"


def test_keys_are_safe_whatever_the_name() -> None:
    report = EvalReport(name="../../etc ☃ " * 10, started_at="", wall_s=0.0, cases=())
    key = report_key(report)
    assert REPORT_NAME.match(key + ".json") and key.startswith("report-etc-")
    assert report_key(report.model_copy(update={"name": "☃"})).startswith("report-eval-")


def test_a_failed_write_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = run_eval_sync(answer, CASES, [exact()])

    def refuse(src: object, dst: object) -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError):
        save_report(report, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_run_eval_saves_when_asked_and_checks_the_folder_first(tmp_path: Path) -> None:
    report = run_eval_sync(answer, CASES, [exact()], save_to=tmp_path)
    (saved,) = tmp_path.iterdir()
    assert EvalReport.from_json(saved.read_bytes()).cases == report.cases
    with pytest.raises(ConfigError, match="save_to must be a folder path"):
        run_eval_sync(answer, CASES, [exact()], save_to=3)
    assert DEFAULT_EVALS_FOLDER == ".agent_adapter/evals"


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A module with an agent and a cases file in a fresh current folder."""
    name = f"saveapp_{uuid.uuid4().hex[:8]}"
    (tmp_path / f"{name}.py").write_text("def answer(q):\n    return 'shipped'\n", "utf-8")
    case = {"id": "a", "input": "1042", "expected": "shipped"}
    (tmp_path / "cases.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield name
    sys.modules.pop(name, None)


def cli(*argv: str) -> tuple[int, str]:
    parser = argparse.ArgumentParser(prog="agent-adapter")
    add_eval_parser(parser.add_subparsers(dest="command", required=True))
    out = io.StringIO()
    code = run_eval_command(parser.parse_args(["eval", *argv, "--no-record"]), out)
    return code, out.getvalue()


def test_the_cli_saves_to_the_default_folder_or_a_named_one(app: str, tmp_path: Path) -> None:
    code, text = cli(f"{app}:answer", "cases.jsonl", "--save")
    (saved,) = (tmp_path / DEFAULT_EVALS_FOLDER).iterdir()
    assert code == 0 and f"saved {saved.relative_to(tmp_path)}" in text
    cli(f"{app}:answer", "cases.jsonl", "--save", "mine")
    assert len(list((tmp_path / "mine").iterdir())) == 1
    assert "saved" not in cli(f"{app}:answer", "cases.jsonl")[1]
