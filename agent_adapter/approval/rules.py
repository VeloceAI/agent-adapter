"""Proposed confirm rules from past decisions; a person accepts each one before it applies.

A rule covers every non-secret top-level argument of the approvals it rests on, bounded to
the values seen there, and asks for anything else: an unseen value, a field it does not
cover, a missing field, or any call after its review date.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from agent_adapter.approval.memory import decided
from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import ToolCall
from agent_adapter.hooks.builtin.logging import is_secret_field

Scalar = str | int | float | bool
Op = Literal["==", "in", "between"]
MAX_NAME = 120
MAX_SAMPLE = 5
MAX_CHOICES = 10  # distinct values a non-numeric field may take and still be bounded
DAY_S = 86_400.0
_CONFIG_KEYS = frozenset({"tool", "conditions", "review_by"})
_CONDITION_KEYS = frozenset({"field", "op", "values"})
_FROZEN = ConfigDict(frozen=True)


class Condition(BaseModel):
    """One argument bounded to the evidence: `==` one value, `in` a set, `between` lo..hi."""

    model_config = _FROZEN

    field: str
    op: Op
    values: tuple[Scalar, ...]

    def holds(self, arguments: Mapping[str, Any]) -> bool:
        """True only when the field is present, of a seen type, and inside the bounds."""
        if self.field not in arguments:
            return False
        given = arguments[self.field]
        if self.op == "between":
            low, high = self.values
            return _number(given) and low <= given <= high
        return any(type(given) is type(value) and given == value for value in self.values)

    def describe(self) -> str:
        """The condition as text; names and values are quoted, so they cannot break a line."""
        if self.op == "between":
            return f"{self.values[0]!r} <= {self.field!r} <= {self.values[1]!r}"
        if self.op == "==":
            return f"{self.field!r} == {self.values[0]!r}"
        return f"{self.field!r} in {self.values!r}"


class RuleEvidence(BaseModel):
    """The decisions a proposal rests on, and those against it."""

    model_config = _FROZEN

    approved: int
    denied: int  # denials and edited approvals inside the bounds (counter-evidence)
    denied_outside: int  # other decisions on the tool: a reason the bounds stop where they do
    deciders: tuple[str, ...]
    first_decided_at: float
    last_decided_at: float
    sample: tuple[str, ...]  # approval ids


class ProposedRule(BaseModel):
    """Data only: skip asking for `tool` when every condition holds; inert until `accept`."""

    model_config = _FROZEN

    tool: str
    conditions: tuple[Condition, ...]
    evidence: RuleEvidence
    review_days: float

    def describe(self) -> str:
        """One line: the rule and its evidence."""
        e = self.evidence
        terms = " and ".join(c.describe() for c in self.conditions) or "no arguments"
        return (
            f"skip asking for {self.tool!r} only when {terms} "
            f"({e.approved} approved by {len(e.deciders)}, {e.denied} against, "
            f"{e.denied_outside} denied outside; review in {self.review_days:g} days)"
        )


@dataclass(frozen=True)
class RuleConfirm:
    """A `confirm=` callable: False (no ask) only for `tool`, every field covered and in bounds,
    before `review_by`; True (ask) for anything else."""

    tool: str
    conditions: tuple[Condition, ...]
    review_by: float
    clock: Callable[[], float] = field(default=time.time, compare=False)

    def __call__(self, call: ToolCall, ctx: ToolContext) -> bool:
        if call.name != self.tool or self.clock() > self.review_by:
            return True
        covered = {c.field for c in self.conditions}
        if set(_open_fields(call.arguments)) != covered:
            return True  # a field the rule never saw, or one it covers is missing
        return not all(c.holds(call.arguments) for c in self.conditions)


@dataclass(frozen=True)
class AcceptedRule:
    """A rule a person accepted: commit `code` (or `config`) and pass `confirm` as `confirm=`."""

    rule: ProposedRule
    accepted_by: str
    accepted_at: float
    config: dict[str, Any]
    code: str
    confirm: RuleConfirm


def propose_rules(
    records: Iterable[ApprovalRecord],
    *,
    min_count: int = 5,
    min_deciders: int = 2,
    max_denied: int = 0,
    review_days: float = 90.0,
    tools: Iterable[str] | None = None,
) -> list[ProposedRule]:
    """Rules the approvals support: one per tool and argument shape, bounded to the seen values.

    Only tool approvals decided as-is count as evidence. A shape whose fields cannot all be
    bounded (a nested value, or a text field with over `MAX_CHOICES` values) gets no rule.

    Raises:
        ConfigError: `min_count` < 2, `min_deciders` < 1, `max_denied` < 0 or `review_days` <= 0.
    """
    for label, value, low in (("min_count", min_count, 2), ("min_deciders", min_deciders, 1)):
        _check_int(label, value, low)
    _check_int("max_denied", max_denied, 0)
    if not _number(review_days) or review_days <= 0:
        raise ConfigError(f"review_days must be a number > 0, got {review_days!r}")
    wanted = None if tools is None else set(tools)
    by_tool: dict[str, list[ApprovalRecord]] = defaultdict(list)
    for record in decided(records):
        request = record.request
        if request.kind == "tool" and (wanted is None or request.tool in wanted):
            by_tool[request.tool].append(record)
    found: list[ProposedRule] = []
    for tool, items in sorted(by_tool.items()):
        shapes: dict[frozenset[str], list[ApprovalRecord]] = defaultdict(list)
        for record in items:
            if _approved_as_is(record):
                shapes[frozenset(_open_fields(record.request.arguments))].append(record)
        for group in shapes.values():
            rule = _rule(tool, group, items, min_count, min_deciders, max_denied, review_days)
            if rule is not None:
                found.append(rule)
    found.sort(key=lambda rule: (-rule.evidence.approved, rule.tool))
    return found


def _check_int(label: str, value: object, low: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ConfigError(f"{label} must be an int >= {low}, got {value!r}")


def _open_fields(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The top-level arguments that are not secret-named."""
    return {k: v for k, v in arguments.items() if not is_secret_field(k, v)}


def _approved_as_is(record: ApprovalRecord) -> bool:
    decision = record.decision
    return decision is not None and decision.approved and decision.edited_args is None


def _bounds(name: str, values: list[Any]) -> Condition | None:
    """The tightest condition over these values, or None when they cannot be bounded."""
    if all(_number(v) for v in values) and (
        len(set(values)) > 1 or any(isinstance(v, float) for v in values)
    ):
        return Condition(field=name, op="between", values=(min(values), max(values)))
    if not all(isinstance(v, Scalar) and (not isinstance(v, float) or _number(v)) for v in values):
        return None  # nested, null or non-finite: not bounded
    seen: list[Scalar] = []
    for value in values:
        if not any(type(value) is type(s) and value == s for s in seen):
            seen.append(value)
    if len(seen) > MAX_CHOICES:
        return None
    if len(seen) == 1:
        return Condition(field=name, op="==", values=(seen[0],))
    seen.sort(key=lambda value: (type(value).__name__, value))
    return Condition(field=name, op="in", values=tuple(seen))


def _rule(
    tool: str,
    group: list[ApprovalRecord],
    everything: list[ApprovalRecord],
    min_count: int,
    min_deciders: int,
    max_denied: int,
    review_days: float,
) -> ProposedRule | None:
    deciders = sorted(
        {r.decision.decided_by for r in group if r.decision and r.decision.decided_by}
    )
    if len(group) < min_count or len(deciders) < min_deciders:
        return None
    names = sorted(_open_fields(group[0].request.arguments))
    conditions = []
    for name in names:
        condition = _bounds(name, [r.request.arguments[name] for r in group])
        if condition is None:
            return None
        conditions.append(condition)
    probe = RuleConfirm(tool, tuple(conditions), math.inf)
    others = [r for r in everything if not _approved_as_is(r)]
    inside = sum(1 for r in others if not probe(_call(r), ToolContext(run_id="")))
    if inside > max_denied:
        return None
    stamps = [r.decision.decided_at for r in group if r.decision is not None]
    evidence = RuleEvidence(
        approved=len(group),
        denied=inside,
        denied_outside=len(others) - inside,
        deciders=tuple(deciders),
        first_decided_at=min(stamps),
        last_decided_at=max(stamps),
        sample=tuple(r.approval_id for r in group[:MAX_SAMPLE]),
    )
    return ProposedRule(
        tool=tool, conditions=tuple(conditions), evidence=evidence, review_days=review_days
    )


def _call(record: ApprovalRecord) -> ToolCall:
    request = record.request
    return ToolCall(name=request.tool, arguments=request.arguments, id=request.call_id)


def accept(rule: ProposedRule, *, accepted_by: str) -> AcceptedRule:
    """A person's explicit acceptance: config and code to commit, and the `confirm=` callable.

    The rule stops skipping at `accepted_at + review_days`; after that every call asks again.

    Raises:
        ConfigError: not a ProposedRule, or `accepted_by` empty, too long or not printable.
    """
    if not isinstance(rule, ProposedRule):
        raise ConfigError(f"accept needs a ProposedRule, got {type(rule).__name__}")
    name = str(accepted_by or "").strip()
    if not name or len(name) > MAX_NAME or not name.isprintable():
        raise ConfigError(f"accepted_by must name a person (1-{MAX_NAME} printable characters)")
    now = time.time()
    review_by = round(now + rule.review_days * DAY_S, 3)
    config = {
        "tool": rule.tool,
        "conditions": [
            {"field": c.field, "op": c.op, "values": list(c.values)} for c in rule.conditions
        ],
        "review_by": review_by,
    }
    code = (
        "from agent_adapter.approval import confirm_from_config\n\n"
        f"# accepted by {name!r}: {rule.describe()}\n"
        f"confirm = confirm_from_config({config!r})\n"
    )
    return AcceptedRule(rule, name, now, config, code, confirm_from_config(config))


def confirm_from_config(config: Mapping[str, Any]) -> RuleConfirm:
    """The `confirm=` callable of a committed rule config (`AcceptedRule.config`).

    Raises:
        ConfigError: other keys, a bad condition, a secret-named field or a bad `review_by`.
    """
    if not isinstance(config, Mapping) or set(config) != _CONFIG_KEYS:
        raise ConfigError(f"a rule config has exactly the keys {sorted(_CONFIG_KEYS)}")
    tool, items, review_by = config["tool"], config["conditions"], config["review_by"]
    if not isinstance(tool, str) or not tool:
        raise ConfigError("a rule config needs a non-empty tool")
    if not _number(review_by):
        raise ConfigError("review_by must be a finite timestamp")
    if not isinstance(items, list):
        raise ConfigError("conditions must be a list")
    conditions = tuple(_condition(item) for item in items)
    if len({c.field for c in conditions}) != len(conditions):
        raise ConfigError("a rule tests each field once")
    return RuleConfirm(tool, conditions, float(review_by))


def _condition(item: object) -> Condition:
    if not isinstance(item, Mapping) or set(item) != _CONDITION_KEYS:
        raise ConfigError(f"a condition has exactly the keys {sorted(_CONDITION_KEYS)}")
    name, op, values = item["field"], item["op"], item["values"]
    if not isinstance(name, str) or not name or not isinstance(values, list | tuple) or not values:
        raise ConfigError("a condition needs a field and a non-empty list of values")
    if is_secret_field(name, values[0]):
        raise ConfigError("a rule may not test a secret-named field")
    if op == "between":
        if len(values) != 2 or not all(_number(v) for v in values) or values[0] > values[1]:
            raise ConfigError("'between' needs two finite numbers, low first")
    elif op in ("==", "in"):
        if (op == "==") != (len(values) == 1) or len(values) > MAX_CHOICES:
            raise ConfigError("'==' takes one value, 'in' 2 to 10")
        if not all(isinstance(v, str | int | bool) for v in values):
            raise ConfigError("'==' and 'in' take str, int or bool values")
    else:
        raise ConfigError("op must be '==', 'in' or 'between'")
    return Condition(field=name, op=op, values=tuple(values))


def _number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value)
