"""Ready-made graders `async (case, run) -> Score | None`; they never raise (an error is None)."""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.evals._grading import Grader, expected_values, guarded, verdict
from agent_adapter.evals._schema_check import schema_problems
from agent_adapter.evals._text import normalised, output_text
from agent_adapter.evals.judge import JUDGE_SCHEMA, LLMJudge, llm_judge
from agent_adapter.evals.model import EvalCase, Score

logger = logging.getLogger(__name__)

CheckResult = bool | float | Score
Check = Callable[[EvalCase, Any], CheckResult | Awaitable[CheckResult]]
_TOOL_METRIC = "tool_calls."  # the optimiser's per-tool call counts


def exact(*, normalise: bool = True) -> Grader:
    """1.0 when the output equals the expected text (or one of a list of them)."""
    norm = normalised if normalise else str

    async def grade(case: EvalCase, run: Any) -> Score | None:
        wanted = expected_values(case)
        if wanted is None:
            return None
        hit = norm(output_text(run)) in {norm(w) for w in wanted}
        return verdict(hit, "not the expected answer")

    return guarded("exact", grade)


def contains(*, normalise: bool = True) -> Grader:
    """1.0 when the expected text (or one of a list of them) appears in the output."""
    norm = normalised if normalise else str

    async def grade(case: EvalCase, run: Any) -> Score | None:
        wanted = expected_values(case)
        if wanted is None:
            return None
        output = norm(output_text(run))
        return verdict(any(norm(w) in output for w in wanted), "the expected answer is missing")

    return guarded("contains", grade)


def regex(pattern: str | None = None, *, flags: int = re.IGNORECASE) -> Grader:
    """1.0 when `pattern` (default: the expected pattern or list of them) matches the output.

    Raises:
        ConfigError: `pattern` is not a valid regular expression.
    """
    fixed: re.Pattern[str] | None = None
    if pattern is not None:
        try:
            fixed = re.compile(pattern, flags)
        except re.error as exc:
            raise ConfigError(f"regex: invalid pattern ({exc})") from exc

    async def grade(case: EvalCase, run: Any) -> Score | None:
        text = output_text(run)
        if fixed is not None:
            hit = fixed.search(text) is not None
        else:
            patterns = expected_values(case)
            if patterns is None:
                return None
            hit = any(re.search(p, text, flags) for p in patterns)
        return verdict(hit, "the output does not match the expected pattern")

    return guarded("regex", grade)


def json_schema(schema: Mapping[str, Any]) -> Grader:
    """1.0 when the output (JSON text, a dict or a model) matches `schema`.

    Checked keywords: type, enum, const, bounds, lengths, properties, required,
    additionalProperties, items, anyOf and oneOf; others are ignored.

    Raises:
        ConfigError: `schema` is not a JSON object.
    """
    if not isinstance(schema, Mapping):
        raise ConfigError(f"json_schema needs a schema object, got {type(schema).__name__}")
    frozen = json.loads(json.dumps(schema))

    async def grade(case: EvalCase, run: Any) -> Score | None:
        value = getattr(run, "output", None)
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        elif isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return verdict(False, "the output is not JSON")
        problems = schema_problems(value, frozen)
        return verdict(not problems, "; ".join(problems[:3]))

    return guarded("json_schema", grade)


def tools_called(*names: str, forbid: Iterable[str] = (), only: bool = False) -> Grader:
    """Every tool in `names` was called, none in `forbid` was, and with `only` no other one.

    `value` is the share of `names` called (0 when a forbidden or extra tool ran).

    Raises:
        ConfigError: no names and nothing forbidden, or a name that is not text.
    """
    required, banned = tuple(names), frozenset(forbid)
    if not required and not banned:
        raise ConfigError("tools_called needs tool names or forbid=")
    if not all(isinstance(n, str) and n for n in (*required, *banned)):
        raise ConfigError("tools_called: tool names must be non-empty text")

    async def grade(case: EvalCase, run: Any) -> Score | None:
        called = _tools_of(run)
        if called is None:
            logger.warning("tools_called on %s: the run has no tool calls recorded", case.id)
            return None
        missing = [n for n in required if n not in called]
        bad = sorted(banned & called)
        extra = sorted(called - set(required)) if only else []
        share = (len(required) - len(missing)) / len(required) if required else 1.0
        value = 0.0 if bad or extra else share
        notes = "; ".join(
            text
            for text in (
                f"not called: {', '.join(missing)}" if missing else "",
                f"forbidden: {', '.join(bad)}" if bad else "",
                f"not allowed: {', '.join(extra)}" if extra else "",
            )
            if text
        )
        return Score(value=value, passed=not (missing or bad or extra), notes=notes)

    return guarded("tools_called", grade)


def max_cost(usd: float) -> Grader:
    """Passes when the run cost at most `usd` (from the ledger, else the events).

    Raises:
        ConfigError: `usd` is not a finite number >= 0.
    """
    limit = _limit("max_cost", usd)

    async def grade(case: EvalCase, run: Any) -> Score | None:
        spent = _measure(run, "cost_usd")
        if spent is None:
            return None
        return verdict(spent <= limit, f"cost ${spent:.6f} is over ${limit:.6f}")

    return guarded("max_cost", grade)


def max_latency(seconds: float) -> Grader:
    """Passes when the run took at most `seconds` of wall time.

    Raises:
        ConfigError: `seconds` is not a finite number >= 0.
    """
    limit = _limit("max_latency", seconds)

    async def grade(case: EvalCase, run: Any) -> Score | None:
        took = _measure(run, "latency_s")
        if took is None:
            return None
        return verdict(took <= limit, f"took {took:.3f}s, over {limit:.3f}s")

    return guarded("max_latency", grade)


def code_check(fn: Check) -> Grader:
    """Your check: `fn(case, run)`, sync or async, returns a bool, a float in [0, 1] or a Score."""

    async def grade(case: EvalCase, run: Any) -> Score | None:
        value = fn(case, run)
        if inspect.isawaitable(value):
            value = await value
        if isinstance(value, Score):
            return value
        if isinstance(value, bool):
            return verdict(value, "the check failed")
        if isinstance(value, int | float) and math.isfinite(value) and 0 <= value <= 1:
            return Score(value=float(value))
        logger.warning("code_check on %s returned %s; not graded", case.id, type(value).__name__)
        return None

    return guarded("code_check", grade)


def all_of(*graders: Grader, weights: Sequence[float] | None = None) -> Grader:
    """Weighted mean of the graders that graded; passed only if every one graded and none failed.

    A sub-grader that errors or does not grade makes `passed` False (fail closed); `passed` is
    None only when every sub-grader gave a value without a verdict.

    Raises:
        ConfigError: no graders, or weights of the wrong length, negative or summing to 0.
    """
    if not graders:
        raise ConfigError("all_of needs at least one judge")
    shares = [1.0] * len(graders) if weights is None else [float(w) for w in weights]
    if len(shares) != len(graders):
        raise ConfigError(f"all_of got {len(graders)} judges but {len(shares)} weights")
    if any(not math.isfinite(w) or w < 0 for w in shares) or sum(shares) <= 0:
        raise ConfigError("all_of weights must be finite, >= 0 and not all 0")
    raw: list[Grader] = [getattr(g, "_unguarded", g) for g in graders]

    async def grade(case: EvalCase, run: Any) -> Score | None:
        graded: list[tuple[Score, float]] = []
        missing: list[str] = []  # a required check that did not grade cannot count as passed
        for index, (one, share) in enumerate(zip(raw, shares, strict=True)):
            try:
                score = await one(case, run)
            except Exception as exc:
                if is_control_flow(exc):
                    raise
                log_failure(
                    logger, logging.WARNING, exc, "all_of[%d] grader failed on %s", index, case.id
                )
                missing.append(f"all_of[{index}] failed to grade")
                continue
            if isinstance(score, Score):
                graded.append((score, share))
                continue
            if score is not None:
                logger.warning("all_of: a judge returned %s on %s", type(score).__name__, case.id)
            missing.append(f"all_of[{index}] not graded")
        total = sum(share for _, share in graded)
        if total <= 0:
            return None
        verdicts = [s.passed for s, _ in graded if s.passed is not None]
        if missing or False in verdicts:
            passed: bool | None = False
        else:
            passed = True if verdicts else None
        return Score(
            value=min(sum(s.value * share for s, share in graded) / total, 1.0),
            passed=passed,
            notes="; ".join([*(s.notes for s, _ in graded if s.notes), *missing]),
        )

    grade.__name__ = grade.__qualname__ = "all_of"
    grade.quotes_output = any(getattr(g, "quotes_output", False) for g in graders)  # type: ignore[attr-defined]
    grade.parts = graders  # type: ignore[attr-defined]  # lets a caller rebuild it
    grade.weights = tuple(shares)  # type: ignore[attr-defined]
    return grade


def _tools_of(run: Any) -> frozenset[str] | None:
    """Tool names the run called: `run.tool_calls` (None: not observed), else per-tool metrics."""
    if hasattr(run, "tool_calls"):
        calls = run.tool_calls
        return None if calls is None else frozenset(str(name) for name in calls)
    metrics = getattr(run, "metrics", None)
    if isinstance(metrics, Mapping):
        return frozenset(
            key[len(_TOOL_METRIC) :]
            for key, count in metrics.items()
            if key.startswith(_TOOL_METRIC) and count
        )
    return None


def _measure(run: Any, name: str) -> float | None:
    """`run.<name>`, else `run.metrics[name]`; None (logged) when neither is a number."""
    value = getattr(run, name, None)
    if value is None and not hasattr(run, name):  # None on an EvalRun means "not observed"
        metrics = getattr(run, "metrics", None)
        value = metrics.get(name) if isinstance(metrics, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        logger.warning("the run has no %s; not graded", name)
        return None
    return float(value)


def _limit(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ConfigError(f"{name} needs a finite number >= 0, got {value!r}")
    return float(value)


__all__ = [
    "JUDGE_SCHEMA",
    "Check",
    "Grader",
    "LLMJudge",
    "all_of",
    "code_check",
    "contains",
    "exact",
    "json_schema",
    "llm_judge",
    "max_cost",
    "max_latency",
    "regex",
    "tools_called",
]
