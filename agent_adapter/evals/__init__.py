"""Evals: run cases on any agent, grade the runs, and report for people and CI.

Names load on first use, so the pytest plugin can import this package cheaply.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_adapter.evals import graders
    from agent_adapter.evals.cases import load_cases
    from agent_adapter.evals.from_runs import (
        RecordedRun,
        RunCase,
        RunCases,
        SkippedRun,
        cases_from_runs,
        write_cases,
    )
    from agent_adapter.evals.graders import Grader, LLMJudge
    from agent_adapter.evals.model import EvalCase, EvalRun, Score
    from agent_adapter.evals.report import (
        CaseChange,
        CaseResult,
        EvalComparison,
        EvalReport,
        GradeResult,
        GroupSummary,
    )
    from agent_adapter.evals.runner import (
        CASE_EVENT,
        assert_eval,
        eval_executor,
        run_eval,
        run_eval_sync,
    )
    from agent_adapter.evals.store import DEFAULT_EVALS_FOLDER, save_report
    from agent_adapter.evals.targets import (
        CallableTarget,
        ChatTarget,
        FlowTarget,
        Outcome,
        Target,
        ToolLoopTarget,
        as_target,
    )

_HOMES = {
    "EvalCase": "model",
    "EvalRun": "model",
    "Score": "model",
    "load_cases": "cases",
    "RecordedRun": "from_runs",
    "RunCase": "from_runs",
    "RunCases": "from_runs",
    "SkippedRun": "from_runs",
    "cases_from_runs": "from_runs",
    "write_cases": "from_runs",
    "Grader": "graders",
    "LLMJudge": "graders",
    "CaseChange": "report",
    "CaseResult": "report",
    "EvalComparison": "report",
    "EvalReport": "report",
    "GradeResult": "report",
    "GroupSummary": "report",
    "CASE_EVENT": "runner",
    "assert_eval": "runner",
    "eval_executor": "runner",
    "run_eval": "runner",
    "run_eval_sync": "runner",
    "DEFAULT_EVALS_FOLDER": "store",
    "save_report": "store",
    "CallableTarget": "targets",
    "ChatTarget": "targets",
    "FlowTarget": "targets",
    "Outcome": "targets",
    "Target": "targets",
    "ToolLoopTarget": "targets",
    "as_target": "targets",
}


def _module(home: str) -> ModuleType | None:
    """The submodule named `home`, imported now; None for an unknown home."""
    if home == "graders":
        import agent_adapter.evals.graders

        return agent_adapter.evals.graders
    if home == "model":
        import agent_adapter.evals.model

        return agent_adapter.evals.model
    if home == "cases":
        import agent_adapter.evals.cases

        return agent_adapter.evals.cases
    if home == "from_runs":
        import agent_adapter.evals.from_runs

        return agent_adapter.evals.from_runs
    if home == "report":
        import agent_adapter.evals.report

        return agent_adapter.evals.report
    if home == "runner":
        import agent_adapter.evals.runner

        return agent_adapter.evals.runner
    if home == "store":
        import agent_adapter.evals.store

        return agent_adapter.evals.store
    if home == "targets":
        import agent_adapter.evals.targets

        return agent_adapter.evals.targets
    return None


def __getattr__(name: str) -> Any:
    """Import a public name on first use.

    Raises:
        AttributeError: not a public name of this package.
    """
    module = _module("graders" if name == "graders" else _HOMES.get(name, ""))
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = module if name == "graders" else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "CASE_EVENT",
    "DEFAULT_EVALS_FOLDER",
    "CallableTarget",
    "CaseChange",
    "CaseResult",
    "ChatTarget",
    "EvalCase",
    "EvalComparison",
    "EvalReport",
    "EvalRun",
    "FlowTarget",
    "GradeResult",
    "Grader",
    "GroupSummary",
    "LLMJudge",
    "Outcome",
    "RecordedRun",
    "RunCase",
    "RunCases",
    "Score",
    "SkippedRun",
    "Target",
    "ToolLoopTarget",
    "as_target",
    "assert_eval",
    "cases_from_runs",
    "eval_executor",
    "graders",
    "load_cases",
    "run_eval",
    "run_eval_sync",
    "save_report",
    "write_cases",
]
