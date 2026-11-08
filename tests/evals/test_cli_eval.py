import argparse
import io
import json
import sys
import textwrap
import uuid
import xml.etree.ElementTree as ET

import pytest

from agent_adapter.cli.eval import add_eval_parser, parse_grader, run_eval_command
from agent_adapter.core.errors import ConfigError
from agent_adapter.evals import EvalReport

APP = """
from agent_adapter.core.registry import Registry
from agent_adapter.evals import Score
from agent_adapter.exec.executor import Executor

ANSWERS = {"1042": "shipped", "1043": "packed"}


def answer(question):
    return ANSWERS.get(question, "unknown")


def broken(question):
    raise RuntimeError("down")


async def polite(case, run):
    return Score(value=1.0, passed=True, notes="polite")


executor = Executor(Registry())
not_a_grader = 3
"""


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A module of agents in a fresh folder, imported from the current directory."""
    name = f"evalapp_{uuid.uuid4().hex[:8]}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(APP), encoding="utf-8")
    cases = [
        {"id": "shipped", "input": "1042", "expected": "shipped", "tags": ["orders"]},
        {"id": "packed", "input": "1043", "expected": "packed", "tags": ["orders"]},
    ]
    (tmp_path / "cases.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield name
    sys.modules.pop(name, None)


def parse(*argv):
    parser = argparse.ArgumentParser(prog="agent-adapter")
    commands = parser.add_subparsers(dest="command", required=True)
    add_eval_parser(commands)
    return parser.parse_args(["eval", *argv])


def run(*argv):
    out = io.StringIO()
    code = run_eval_command(parse(*argv), out)
    return code, out.getvalue()


def test_a_passing_eval_exits_0_and_writes_reports(app, tmp_path):
    code, text = run(
        f"{app}:answer",
        "cases.jsonl",
        "--grader",
        "exact",
        "--grader",
        f"{app}:polite",
        "--junit",
        "out/evals.xml",
        "--markdown",
        "out/evals.md",
        "--no-record",
    )
    assert code == 0, text
    assert text.startswith("eval answer: 2/2 passed (min 100%), mean score 1.000 (min 1.000)")
    suite = ET.fromstring(
        (tmp_path / "out/evals.xml").read_text(encoding="utf-8").split("\n", 1)[1]
    )
    assert suite.find("testsuite").attrib["tests"] == "2"
    assert "# Eval: answer" in (tmp_path / "out/evals.md").read_text(encoding="utf-8")


def test_below_the_bar_exits_1_and_names_the_failing_case(app):
    code, text = run(f"{app}:answer", "cases.jsonl", "--grader", "regex=^ship", "--no-record")
    assert code == 1
    assert "FAIL packed (0.000) run run_" in text and "regex: the output does not match" in text
    lenient, _ = run(
        f"{app}:answer",
        "cases.jsonl",
        "--grader",
        "regex=^ship",
        "--min-score",
        "0.5",
        "--min-pass-rate",
        "0.5",
        "--no-record",
    )
    assert lenient == 0
    strict, _ = run(
        f"{app}:answer",
        "cases.jsonl",
        "--grader",
        "regex=^ship",
        "--min-score",
        "0.5",
        "--no-record",
    )
    assert strict == 1  # the default --min-pass-rate 1.0 still wants every case to pass


def test_an_erroring_case_exits_1_even_above_the_bar(app):
    code, text = run(f"{app}:broken", "cases.jsonl", "--min-score", "0", "--no-record")
    assert code == 1 and "internal-error: RuntimeError" in text


def test_json_goes_to_stdout_and_the_summary_to_stderr(app, capsys):
    code, text = run(f"{app}:answer", "cases.jsonl", "--json", "--no-record", "--name", "shop")
    report = EvalReport.from_json(text)
    assert code == 0 and report.name == "shop" and report.n_passed == 2
    assert "eval shop: 2/2 passed" in capsys.readouterr().err


def test_a_baseline_regression_exits_1(app, tmp_path):
    _, baseline = run(f"{app}:answer", "cases.jsonl", "--json", "--no-record")
    (tmp_path / "base.json").write_text(baseline, encoding="utf-8")
    worse, text = run(
        f"{app}:answer",
        "cases.jsonl",
        "--grader",
        "regex=^ship",
        "--min-score",
        "0",
        "--min-pass-rate",
        "0",
        "--baseline",
        "base.json",
        "--no-record",
    )
    assert worse == 1 and "1 regressed" in text and "REGRESSED packed" in text
    same, _ = run(f"{app}:answer", "cases.jsonl", "--baseline", "base.json", "--no-record")
    assert same == 0


def test_runs_are_recorded_by_default(app, tmp_path):
    code, _ = run(f"{app}:answer", "cases.jsonl", "--runs", "runs")
    assert code == 0 and list((tmp_path / "runs").rglob("*.jsonl"))
    mine, _ = run(
        f"{app}:answer", "cases.jsonl", "--executor", f"{app}:executor", "--runs", "runs2"
    )
    assert mine == 0 and list((tmp_path / "runs2").rglob("*.jsonl"))


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["{app}:missing", "cases.jsonl"], "has no 'missing'"),
        (["{app}:answer", "nope.jsonl"], "cannot read it"),
        (["{app}:answer", "cases.jsonl", "--baseline", "nope.json"], "cannot read it"),
        (["{app}:answer", "cases.jsonl", "--baseline", "cases.jsonl"], "not an eval report"),
        (["{app}:answer", "cases.jsonl", "--executor", "{app}:not_a_grader"], "an Executor"),
        (["{app}:answer", "cases.jsonl", "--grader", "{app}:not_a_grader"], "is not a grader"),
    ],
)
def test_bad_inputs_are_config_errors(app, argv, message):
    with pytest.raises(ConfigError, match=message):
        run(*[a.format(app=app) for a in argv], "--no-record")
    with pytest.raises(ConfigError, match=message):
        run(*[a.format(app=app) for a in argv], "--runs", "runs")


def test_grader_specs(tmp_path):
    (tmp_path / "schema.json").write_text('{"type": "object"}', encoding="utf-8")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    for spec, name in [
        ("exact", "exact"),
        ("contains", "contains"),
        ("regex", "regex"),
        ("regex=a+", "regex"),
        (f"json-schema={tmp_path / 'schema.json'}", "json_schema"),
        ("tools=a, b", "tools_called"),
        ("max-cost=0.5", "max_cost"),
        ("max-latency=2", "max_latency"),
    ]:
        assert parse_grader(spec).__name__ == name
    for spec, message in [
        ("tools=", "needs a value"),
        ("max-cost=x", "not a number"),
        ("max-cost=-1", ">= 0"),
        (f"json-schema={tmp_path / 'none.json'}", "cannot read it"),
        (f"json-schema={tmp_path / 'bad.json'}", "not valid JSON"),
        ("nonsense", "use module.path:attribute"),
    ]:
        with pytest.raises(ConfigError, match=message):
            parse_grader(spec)


@pytest.mark.parametrize(
    "argv",
    [["--min-score", "2"], ["--min-score", "x"], ["--concurrency", "0"], ["--timeout", "0"]],
)
def test_bad_numbers_are_usage_errors(argv):
    with pytest.raises(SystemExit):
        parse("app:answer", "cases.jsonl", *argv)


def test_all_ungraded_with_a_full_mean_exits_1(app, tmp_path):
    (tmp_path / "open.jsonl").write_text(
        '{"id": "a", "input": "1042", "expected": null}\n{"id": "b", "input": "1043"}\n',
        encoding="utf-8",
    )
    code, text = run(
        f"{app}:answer",
        "open.jsonl",
        "--grader",
        "contains",
        "--grader",
        "max-cost=1",
        "--min-score",
        "0",
        "--min-pass-rate",
        "0",
        "--no-record",
    )
    assert code == 1 and "0/2 passed" in text  # no case passed: never green


def test_a_missing_baseline_case_exits_1(app, tmp_path):
    _, baseline = run(f"{app}:answer", "cases.jsonl", "--json", "--no-record")
    (tmp_path / "base.json").write_text(baseline, encoding="utf-8")
    (tmp_path / "fewer.jsonl").write_text(
        '{"id": "shipped", "input": "1042", "expected": "shipped"}\n', encoding="utf-8"
    )
    code, text = run(f"{app}:answer", "fewer.jsonl", "--baseline", "base.json", "--no-record")
    assert code == 1 and "MISSING packed" in text


def test_no_content_keeps_case_text_out_of_the_report(app):
    code, text = run(f"{app}:answer", "cases.jsonl", "--json", "--no-content", "--no-record")
    report = EvalReport.from_json(text)
    assert code == 0 and all(c.input == c.expected == c.output == "" for c in report.cases)


def test_a_module_named_like_a_builtin_grader_is_a_user_grader(app, tmp_path, monkeypatch):
    (tmp_path / "tools.py").write_text(
        "from agent_adapter.evals import Score\n\n\nasync def judge(case, run):\n"
        "    return Score(value=1.0, passed=True)\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "tools", raising=False)
    code, _ = run(f"{app}:answer", "cases.jsonl", "--grader", "tools:judge", "--no-record")
    assert code == 0
    sys.modules.pop("tools", None)
