"""Turn recorded runs into eval cases: production runs become regression cases for CI."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent_adapter.core.errors import ConfigError
from agent_adapter.evals import _run_content as content
from agent_adapter.evals._run_files import RecordedRun, read_runs
from agent_adapter.evals.model import EvalCase

logger = logging.getLogger(__name__)

ExpectedFrom = Literal["output", "none"]
MAX_CASE_TAGS = 20
NO_CONTENT = "recorded without capture_content=True, so its input was not kept"
NO_INPUT = "no input was recorded (no flow input, user message or tool arguments)"
CLIPPED = "the recorded input was cut by the recorder (too long), so it is not the real input"
RUNNING = "the run is still running"
UNKNOWN = "no such run"


@dataclass(frozen=True)
class RunCase:
    """An eval case made from one recorded run; `source_run` is that run's id."""

    case: EvalCase
    source_run: str
    run_key: str

    def record(self) -> dict[str, Any]:
        """The JSONL object (`load_cases` reads `id`, `input`, `expected`, `tags`)."""
        return {
            "id": self.case.id,
            "input": self.case.input,
            "expected": self.case.expected,
            "tags": list(self.case.tags),
            "source_run": self.source_run,
            "run_key": self.run_key,
        }


@dataclass(frozen=True)
class SkippedRun:
    """A run that gave no case, and why."""

    run_key: str
    run_id: str
    reason: str


@dataclass(frozen=True)
class RunCases:
    """The cases made from a runs folder, and the runs skipped."""

    cases: tuple[RunCase, ...]
    skipped: tuple[SkippedRun, ...]


def case_from_run(
    run: RecordedRun,
    *,
    expected_from: ExpectedFrom = "output",
    case_id: str = "",
    extra_tags: Iterable[str] = (),
) -> RunCase | SkippedRun:
    """A case from one run's recorded content, or why there is none (never invented data).

    Raises:
        ConfigError: a bad `expected_from`.
    """
    _check_expected(expected_from)
    if run.status == "running":
        return SkippedRun(run.key, run.run_id, RUNNING)
    found = _input(run.events)
    if found is None:
        return SkippedRun(run.key, run.run_id, NO_INPUT if run.capture_content else NO_CONTENT)
    value, output = found
    if content.clipped(value):
        return SkippedRun(run.key, run.run_id, CLIPPED)
    keep = expected_from == "output" and run.status == "ok" and output is not None
    expected = output if keep and not content.clipped(output) else None
    hidden = ["masked"] if content.masked(value) or content.masked(expected) else []
    case = EvalCase(
        id=case_id or f"run-{run.key}",
        input=value,
        expected=expected,
        tags=_tags(run, [*hidden, *extra_tags]),
    )
    return RunCase(case=case, source_run=run.run_id, run_key=run.key)


def cases_from_runs(
    folder: str | os.PathLike[str],
    *,
    run_keys: Collection[str] | None = None,
    status: str | Collection[str] | None = None,
    tags: Collection[str] = (),
    expected_from: ExpectedFrom = "output",
) -> RunCases:
    """Eval cases from the recorded runs in `folder` (a runs folder or one run file).

    `run_keys` picks runs; `status` keeps runs with that status (`ok`, `failed`, `paused`,
    `unfinished`); `tags` keeps runs carrying every one of these run tags. Runs without
    recorded content are skipped with a reason.

    Raises:
        ConfigError: a missing folder, a bad `expected_from`, or a file that is not a run.
        OSError: the folder cannot be listed.
    """
    _check_expected(expected_from)
    wanted = {status} if isinstance(status, str) else set(status or ())
    needed = {tag.strip().lower() for tag in tags}
    keys = set(run_keys) if run_keys is not None else None
    cases: list[RunCase] = []
    skipped: list[SkippedRun] = []
    seen: set[str] = set()
    for run in read_runs(folder):
        seen.add(run.key)
        if keys is not None and run.key not in keys:
            continue
        if (wanted and run.status not in wanted) or not needed <= set(run.tags):
            continue
        made = case_from_run(run, expected_from=expected_from)
        if isinstance(made, RunCase):
            cases.append(made)
        else:
            skipped.append(made)
    skipped += [SkippedRun(key, "", UNKNOWN) for key in sorted((keys or set()) - seen)]
    logger.info("eval cases from runs: %d made, %d runs skipped", len(cases), len(skipped))
    return RunCases(cases=tuple(cases), skipped=tuple(skipped))


def write_cases(path: str | os.PathLike[str], cases: Iterable[RunCase]) -> tuple[int, list[str]]:
    """Append cases to a `.jsonl` file, skipping ids it already holds; (written, skipped ids).

    Raises:
        ConfigError: not a `.jsonl` path, or an existing file that cannot be read.
        OSError: the file cannot be written.
    """
    target = Path(path)
    if target.suffix.lower() != ".jsonl":
        raise ConfigError(f"{target}: write eval cases to a .jsonl file")
    ids = _ids(target)
    lines: list[str] = []
    repeated: list[str] = []
    for found in cases:
        if found.case.id in ids:
            repeated.append(found.case.id)
            continue
        ids.add(found.case.id)
        lines.append(json.dumps(found.record(), ensure_ascii=True, default=str) + "\n")
    if lines:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8", newline="\n") as out:
            out.write("".join(lines))
    logger.info("wrote %d eval case(s) to %s", len(lines), target.name)
    return len(lines), repeated


def _input(events: tuple[dict[str, Any], ...]) -> tuple[Any, Any] | None:
    """(input, recorded output) of a run, or None when no input was recorded."""
    flow = content.flow_io(events)
    if flow is not None:
        return flow
    text = content.first_user_text(events)
    if text is not None:
        return text, content.last_answer_text(events)
    call = content.first_tool_call(events)
    if call is None:
        return None
    call_id, _, arguments = call
    found, result = content.tool_result(events, call_id)
    return arguments, result if found else None


def _tags(run: RecordedRun, extra: Iterable[str]) -> tuple[str, ...]:
    flows = [f"flow:{name}" for name in content.names(run.events, "flow.started", "flow")]
    tools = [f"tool:{name}" for name in content.names(run.events, "tool.started", "tool")]
    every = [*extra, *run.tags, f"status:{run.status}", *flows, *tools]
    return tuple(dict.fromkeys(tag for tag in every if tag))[:MAX_CASE_TAGS]


def _check_expected(expected_from: str) -> None:
    if expected_from not in ("output", "none"):
        raise ConfigError(f'expected_from must be "output" or "none", got {expected_from!r}')


def _ids(target: Path) -> set[str]:
    if not target.exists():
        return set()
    try:
        text = target.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{target}: cannot read it ({exc})") from exc
    ids: set[str] = set()
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            ids.add(str(json.loads(line)["id"]))
        except (ValueError, KeyError, TypeError) as exc:
            raise ConfigError(f"{target}:{number}: not an eval case") from exc
    return ids


__all__ = [
    "ExpectedFrom",
    "RecordedRun",
    "RunCase",
    "RunCases",
    "SkippedRun",
    "case_from_run",
    "cases_from_runs",
    "write_cases",
]
