"""`agent-adapter eval`: run eval cases on an agent, grade them, and fail CI below a score."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, TextIO

from agent_adapter.cli.ui import _import_from_here, _positive_int, _seconds
from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

EPILOG = """graders (repeat --grader; default: contains):
  exact, contains, regex       compare with each case's expected value
  regex=PATTERN                the output must match PATTERN
  json-schema=PATH.json        the output must match the JSON schema in PATH
  tools=NAME[,NAME...]         these tools must be called
  max-cost=USD                 the case may cost at most USD
  max-latency=SECONDS          the case may take at most SECONDS
  module.path:attribute        your own grader, e.g. an LLMJudge built with your model

examples:
  agent-adapter eval app.chat:agent .agent_adapter/chat/eval_cases.jsonl
  agent-adapter eval app:answer cases.jsonl --grader exact --min-score 0.9 --junit evals.xml
  agent-adapter eval app:answer cases.jsonl --save    # then: agent-adapter ui, Evals tab
  agent-adapter eval cases-from-runs .agent_adapter/runs --out cases.jsonl --status failed

TARGET is module.path:attribute, imported from the current folder: a function fn(input),
a chat agent fn(message, session), or an evals Target (FlowTarget, ToolLoopTarget). Exit 1
when no case passes, the pass rate is below --min-pass-rate, the mean score is below
--min-score, a case errors, or --baseline shows a regression or a missing case.

`eval cases-from-runs RUNS_DIR --out FILE.jsonl` turns recorded runs into cases instead
(--status, --tag, --run filter them; --expected-from none leaves `expected` empty). Only
runs recorded with capture_content=True kept an input; the others are listed as skipped."""

_WITH_VALUE = {"regex", "json-schema", "tools", "max-cost", "max-latency"}
FROM_RUNS = "cases-from-runs"
_STATUSES = ("ok", "failed", "paused", "unfinished")


def add_eval_parser(commands: Any) -> None:
    """Register the `eval` subcommand on an argparse subparsers object."""
    from agent_adapter.evals import DEFAULT_EVALS_FOLDER
    from agent_adapter.viewer import DEFAULT_FOLDER

    cmd = commands.add_parser(
        "eval",
        help="run eval cases on an agent, grade them, and fail below a score",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cmd.add_argument("target", help="the agent, module.path:attribute (or cases-from-runs)")
    cmd.add_argument(
        "cases", type=Path, help="a .jsonl, .json or .csv file of cases (or the runs folder)"
    )
    cmd.add_argument(
        "--grader", action="append", default=[], metavar="SPEC", help="a grader (see below)"
    )
    cmd.add_argument("--min-score", type=_share, default=1.0, help="mean score, 0-1 (1.0)")
    cmd.add_argument(
        "--min-pass-rate", type=_share, default=1.0, help="share of cases that pass, 0-1 (1.0)"
    )
    cmd.add_argument("--junit", type=Path, metavar="PATH", help="write a JUnit XML report")
    cmd.add_argument("--markdown", type=Path, metavar="PATH", help="write a Markdown report")
    cmd.add_argument("--json", action="store_true", help="print the report as JSON")
    cmd.add_argument("--baseline", type=Path, metavar="PATH", help="a JSON report to compare")
    cmd.add_argument(
        "--save",
        nargs="?",
        const=DEFAULT_EVALS_FOLDER,
        type=Path,
        metavar="FOLDER",
        help=f"save the JSON report for the viewer's Evals view ({DEFAULT_EVALS_FOLDER})",
    )
    cmd.add_argument("--concurrency", type=_positive_int, default=4, help="cases at once (4)")
    cmd.add_argument("--timeout", type=_seconds, default=300.0, help="seconds per case (300)")
    cmd.add_argument("--executor", metavar="MODULE:ATTR", help="the Executor to run cases on")
    cmd.add_argument("--name", default="", help="the eval's name (default: the target's)")
    cmd.add_argument(
        "--runs", default=DEFAULT_FOLDER, metavar="FOLDER", help=f"record runs ({DEFAULT_FOLDER})"
    )
    cmd.add_argument("--no-record", action="store_true", help="do not record the case runs")
    cmd.add_argument(
        "--no-content",
        action="store_true",
        help="keep no input, expected value, output or judge notes in the reports",
    )
    runs = cmd.add_argument_group("cases-from-runs")
    runs.add_argument("--out", type=Path, metavar="PATH", help="the .jsonl file to append to")
    runs.add_argument(
        "--status", action="append", default=[], choices=_STATUSES, help="keep runs with it"
    )
    runs.add_argument("--tag", action="append", default=[], help="keep runs with this run tag")
    runs.add_argument("--run", action="append", default=[], metavar="KEY", help="only this run")
    runs.add_argument(
        "--expected-from",
        choices=("output", "none"),
        default="output",
        help="output: an ok run's answer is the expected value (default); none: leave it empty",
    )


def run_eval_command(args: argparse.Namespace, out: TextIO) -> int:
    """Run the eval, write the reports, print a summary; 1 below the bar, else 0.

    Raises:
        ConfigError: a bad target, cases file, grader spec, executor or baseline.
        OSError: a report file cannot be written.
    """
    from agent_adapter.evals import (
        EvalReport,
        eval_executor,
        load_cases,
        run_eval_sync,
        save_report,
    )
    from agent_adapter.viewer.chat.targets import load_target

    if args.target == FROM_RUNS:
        return run_cases_from_runs(args, out)
    if args.out or args.status or args.tag or args.run:
        raise ConfigError(f"--out, --status, --tag and --run belong to `eval {FROM_RUNS}`")
    _import_from_here()
    target = load_target(args.target, "TARGET")
    cases = load_cases(args.cases)
    graders = [parse_grader(spec) for spec in args.grader or ["contains"]]
    baseline = _baseline(args.baseline, EvalReport) if args.baseline else None
    own = not args.executor
    default = eval_executor()
    executor = default if own else load_target(args.executor, "--executor")
    if not isinstance(executor, type(default)):  # checked before a recorder attaches to it
        raise ConfigError(
            f"--executor {args.executor!r}: {type(executor).__name__} is not an Executor"
        )
    recorder = None if args.no_record else _record(executor, args.runs)
    try:
        report = run_eval_sync(
            target,
            cases,
            graders,
            concurrency=args.concurrency,
            executor=executor,
            name=args.name,
            timeout_s=args.timeout,
            capture_content=not args.no_content,
        )
    finally:
        if recorder is not None and own:
            recorder.stop()
        elif recorder is not None:
            recorder.flush()  # the caller's executor keeps its recorder
    if args.junit:
        _write(args.junit, report.to_junit())
    if args.markdown:
        _write(args.markdown, report.to_markdown())
    saved = save_report(report, args.save) if args.save is not None else None
    comparison = report.compare(baseline) if baseline is not None else None
    summary = sys.stderr if args.json else out
    if args.json:
        out.write(report.to_json() + "\n")
    summary.write(_summary(report, args, comparison))
    if saved is not None:
        summary.write(f"saved {saved} (agent-adapter ui shows it under Evals)\n")
    below = report.mean_score < args.min_score or report.pass_rate < args.min_pass_rate
    regressed = comparison is not None and comparison.regressed
    return 1 if below or not report.n_passed or report.n_errors or regressed else 0


def run_cases_from_runs(args: argparse.Namespace, out: TextIO) -> int:
    """Append cases made from recorded runs to `--out`; list skipped runs; 1 when none is made.

    Raises:
        ConfigError: no `--out`, a missing runs folder, or an unreadable cases file.
        OSError: the runs folder cannot be listed or the file cannot be written.
    """
    from agent_adapter.evals import cases_from_runs, write_cases

    if args.out is None:
        raise ConfigError(f"eval {FROM_RUNS} needs --out PATH.jsonl")
    found = cases_from_runs(
        args.cases,
        run_keys=args.run or None,
        status=args.status or None,
        tags=args.tag,
        expected_from=args.expected_from,
    )
    written, repeated = write_cases(args.out, found.cases)
    lines = [f"wrote {written} case(s) to {args.out}"]
    lines += [f"  EXISTS {case_id} (kept the saved one)" for case_id in repeated]
    lines += [f"  SKIPPED {s.run_key}: {s.reason}" for s in found.skipped]
    out.write("\n".join(lines) + "\n")
    return 0 if found.cases else 1


def parse_grader(spec: str) -> Any:
    """A grader from a command-line spec (see the `eval` help).

    Raises:
        ConfigError: an unknown spec, a bad value, or a target that is not callable.
    """
    from agent_adapter.evals import graders

    if spec == "exact":
        return graders.exact()
    if spec == "contains":
        return graders.contains()
    if spec == "regex":
        return graders.regex()
    kind, sign, value = spec.partition("=")
    if sign and kind in _WITH_VALUE:
        if not value:
            raise ConfigError(f"--grader {spec!r}: {kind} needs a value after '='")
        if kind == "regex":
            return graders.regex(value)
        if kind == "json-schema":
            return graders.json_schema(_schema(Path(value)))
        if kind == "tools":
            names = [n.strip() for n in value.split(",") if n.strip()]
            return graders.tools_called(*names)
        limit = _number(spec, value)
        return graders.max_cost(limit) if kind == "max-cost" else graders.max_latency(limit)
    from agent_adapter.viewer.chat.targets import load_target

    found = load_target(spec, "--grader")
    if not callable(found):
        raise ConfigError(f"--grader {spec!r}: {type(found).__name__} is not a grader")
    return found


def _summary(report: Any, args: argparse.Namespace, comparison: Any) -> str:
    lines = [
        f"eval {report.name}: {report.n_passed}/{report.n_cases} passed "
        f"(min {args.min_pass_rate:.0%}), "
        f"mean score {report.mean_score:.3f} (min {args.min_score:.3f}), "
        f"errors {report.n_errors}, cost ${report.cost_usd:.4f}"
    ]
    for case in report.failed:
        score = "-" if case.score is None else f"{case.score:.3f}"
        why = case.error or "; ".join(f"{g.grader}: {g.notes}" for g in case.grades if not g.passed)
        lines.append(f"  FAIL {case.case_id} ({score}) run {case.run_id}: {why[:300]}")
    if comparison is not None:
        lines.append(
            f"baseline: score {comparison.score_delta:+.3f}, "
            f"{len(comparison.regressions)} regressed, {len(comparison.improvements)} improved"
        )
        lines += [f"  REGRESSED {change.case_id}" for change in comparison.regressions]
        lines += [f"  MISSING {case_id}" for case_id in comparison.removed]
    return "\n".join(lines) + "\n"


def _record(executor: Any, folder: str) -> Any:
    from agent_adapter.viewer import record

    return record(executor, folder=folder)


def _baseline(path: Path, report_type: Any) -> Any:
    try:
        return report_type.from_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"--baseline {path}: cannot read it ({exc})") from exc
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise ConfigError(f"--baseline {path}: not an eval report") from exc


def _schema(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"json-schema {path}: cannot read it ({exc})") from exc
    except ValueError as exc:
        raise ConfigError(f"json-schema {path}: not valid JSON ({exc})") from exc


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _number(spec: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ConfigError(f"--grader {spec!r}: {text!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise ConfigError(f"--grader {spec!r}: needs a number >= 0")
    return value


def _share(text: str) -> float:
    """An argparse type: a number from 0 to 1."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number from 0 to 1, got {text}") from None
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"must be from 0 to 1, got {text}")
    return value
