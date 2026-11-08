"""Flow state merge rules: how several node updates to one field combine in a superstep."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from agent_adapter.core.errors import FlowDefinitionError, FlowError

S = TypeVar("S", bound=BaseModel)


@runtime_checkable
class MergeRule(Protocol):
    """Combines the updates one superstep makes to a field; `updates` are in deterministic order.

    A rule that adds a write onto the current value (like `append` or `sum`) should also set
    `additive = True`; sub-flows then start the field empty and refuse to map it in and back.
    """

    name: str

    def merge(self, current: Any, updates: Sequence[Any]) -> Any: ...


def _elements(value: Any) -> list[Any]:
    """A list value contributes its elements; anything else contributes itself."""
    return list(value) if isinstance(value, list) else [value]


class _Replace:
    name = "replace"

    def merge(self, current: Any, updates: Sequence[Any]) -> Any:
        first = updates[0]
        if any(value != first for value in updates[1:]):
            raise FlowError("conflicting writes")
        return first


class _Append:
    name = "append"
    additive = True

    def merge(self, current: Any, updates: Sequence[Any]) -> Any:
        merged = list(current or [])
        for value in updates:
            merged.extend(_elements(value))
        return merged


class _Sum:
    name = "sum"
    additive = True

    def merge(self, current: Any, updates: Sequence[Any]) -> Any:
        total = current or 0
        for value in updates:
            total = total + value
        return total


class _Keyed:
    def __init__(self, by: str) -> None:
        self.by = by
        self.name = f"keyed({by})"

    def _key(self, item: Any) -> Any:
        if isinstance(item, Mapping):
            if self.by in item:
                return item[self.by]
        elif hasattr(item, self.by):
            return getattr(item, self.by)
        raise FlowError(f"item without key {self.by!r} written")

    def merge(self, current: Any, updates: Sequence[Any]) -> Any:
        by_key: dict[Any, Any] = {}  # dicts keep first-insertion order; a re-set keeps the slot
        for item in list(current or []):
            by_key[self._key(item)] = item
        for value in updates:
            for item in _elements(value):
                by_key[self._key(item)] = item
        return list(by_key.values())


class Merge:
    """The built-in merge rules; annotate a state field with one (`Annotated[T, Merge.sum]`)."""

    replace: ClassVar[MergeRule] = _Replace()
    append: ClassVar[MergeRule] = _Append()
    sum: ClassVar[MergeRule] = _Sum()

    @staticmethod
    def keyed(by: str) -> MergeRule:
        """A list of mappings or models; a later item with the same `by` key replaces earlier.

        Raises:
            FlowDefinitionError: `by` is empty.
        """
        if not isinstance(by, str) or not by:
            raise FlowDefinitionError([f"keyed(by=...) needs a field name, got {by!r}"])
        return _Keyed(by)


keyed = Merge.keyed


def is_additive(rule: MergeRule) -> bool:
    """True for a rule that adds writes onto the current value (`additive = True`)."""
    return getattr(rule, "additive", False) is True


def merge_rules(model: type[BaseModel]) -> dict[str, MergeRule]:
    """The merge rule of every field of `model`, from `Annotated` metadata (default `replace`).

    Raises:
        FlowDefinitionError: a field carries more than one merge rule.
    """
    rules: dict[str, MergeRule] = {}
    problems: list[str] = []
    for name, info in model.model_fields.items():
        found = [meta for meta in info.metadata if isinstance(meta, MergeRule)]
        if len(found) > 1:
            problems.append(f"field {name!r} has more than one merge rule")
        rules[name] = found[0] if found else Merge.replace
    if problems:
        raise FlowDefinitionError(problems)
    return rules


def apply_updates(
    state: S, updates: Sequence[Mapping[str, Any]], rules: Mapping[str, MergeRule]
) -> S:
    """Merge `updates` (in order) into a copy of `state` and re-validate it.

    Raises:
        FlowError: unknown keys, conflicting `replace` writes, a failing rule, or an invalid result.
    """
    fields = type(state).model_fields
    written: dict[str, list[Any]] = {}
    unknown: set[str] = set()
    for update in updates:
        for key, value in update.items():
            if key in fields:
                written.setdefault(key, []).append(value)
            else:
                unknown.add(key)
    if unknown:
        raise FlowError(f"unknown state fields {sorted(unknown)}")
    if not written:
        return state
    data = dict(state)
    for key, values in written.items():
        rule = rules.get(key, Merge.replace)
        try:
            data[key] = rule.merge(data[key], values)
        except FlowError as exc:
            raise FlowError(f"{exc} to {key}") from exc
        except (TypeError, ValueError) as exc:
            raise FlowError(
                f"cannot merge {key} with rule {rule.name}: {type(exc).__name__}"
            ) from exc
    return validated_state(type(state), data, what="state update")


def validated_state(model: type[S], data: Mapping[str, Any], *, what: str = "state") -> S:
    """`model.model_validate(data)`; the error lists locations and messages, never input values.

    Raises:
        FlowError: `data` does not validate.
    """
    try:
        return model.model_validate(dict(data))
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        ]
        raise FlowError(f"{what} is invalid: {'; '.join(problems)}") from exc
