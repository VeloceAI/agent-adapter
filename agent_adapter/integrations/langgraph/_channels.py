"""Merge rules as LangGraph channels: one channel sees every write of a superstep.

Compiled flows tag their writes (`tagged`) so the order is the flow's, not LangGraph's:
plain writes first (graph input, `Command(update=...)`), then each node's gate edits on their
own, node by node, then every node's updates together in node-name order, item order kept.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

from agent_adapter.core.errors import FlowError, ToolUsageError
from agent_adapter.flow.state import MergeRule
from agent_adapter.integrations.langgraph._integration import INTEGRATION

# marks a tagged write: exactly {WRITE_KEY: [node, phase, seq], "value": v}
WRITE_KEY = "__agent_adapter.langgraph.write/1__"
EDIT, UPDATE = 0, 1  # write phases: gate edits, then node updates


class _Counts:
    """Dicts merged key by key (later writes win); keeps the reserved counters."""

    name = "counts"

    def merge(self, current: Any, updates: Sequence[Any]) -> Any:
        merged = dict(current or {})
        for update in updates:
            merged.update(update)
        return merged


COUNTS: MergeRule = _Counts()


def tagged(node: str, phase: int, seq: int, value: Any) -> dict[str, Any]:
    """A write of `value` by `node`, ordered by (phase, node, seq) when the channel merges."""
    return {WRITE_KEY: [node, phase, seq], "value": value}


def reducer(rule: MergeRule) -> Any:
    """A LangGraph channel applying `rule` to all writes of a superstep: `Annotated[T, reducer(r)]`.

    Raises:
        ToolUsageError: `rule` is not a MergeRule.
        MissingDependency: langgraph is not installed.
    """
    if not isinstance(rule, MergeRule):
        raise ToolUsageError(f"reducer() needs a MergeRule (e.g. Merge.append), got {rule!r}")
    return merge_channel(rule)


def merge_channel(rule: MergeRule) -> Any:
    """A new, empty channel for `rule`."""
    INTEGRATION.require()
    return _channel_class()(rule)


def _tag(value: Any, key: str) -> tuple[str, int, int] | None:
    """The (node, phase, seq) of a tagged write, None for a plain one; FlowError if malformed."""
    if not isinstance(value, Mapping) or WRITE_KEY not in value:
        return None
    tag = value[WRITE_KEY]
    ok = set(value) == {WRITE_KEY, "value"} and isinstance(tag, list | tuple) and len(tag) == 3
    if ok:
        node, phase, seq = tag
        ints = all(isinstance(n, int) and not isinstance(n, bool) for n in (phase, seq))
        ok = isinstance(node, str) and ints and phase in (EDIT, UPDATE) and seq >= 0
    if not ok:
        raise FlowError(f"malformed tagged write to {key}")
    return node, phase, seq


def _groups(values: Sequence[Any], key: str) -> tuple[list[list[Any]], bool]:
    """The writes as merge groups, in order: plain, each node's edits, all node updates."""
    plain: list[Any] = []
    keyed: list[tuple[tuple[int, str, int, int], Any]] = []
    for index, value in enumerate(values):
        tag = _tag(value, key)
        if tag is None:
            plain.append(value)
        else:
            node, phase, seq = tag
            keyed.append(((phase, node, seq, index), value["value"]))
    keyed.sort(key=lambda item: item[0])
    groups: list[list[Any]] = [plain] if plain else []
    current: tuple[int, str] | None = None
    for (phase, node, _seq, _index), value in keyed:
        group = (phase, node if phase == EDIT else "")
        if group != current:
            groups.append([])
            current = group
        groups[-1].append(value)
    return groups, bool(plain)


@functools.cache
def _channel_class() -> type[Any]:
    from langgraph.channels.base import BaseChannel
    from langgraph.channels.last_value import LastValue
    from langgraph.errors import EmptyChannelError

    missing: Any = LastValue(object).checkpoint()  # LangGraph's "empty" sentinel
    base: Any = BaseChannel  # Any: the type check must not depend on the extra

    class MergeChannel(base):  # type: ignore[misc]
        """Holds one state field; `update` merges a superstep's writes with the field's rule."""

        __slots__ = ("rule", "value")

        def __init__(self, rule: MergeRule, typ: Any = Any, key: str = "") -> None:
            super().__init__(typ, key)
            self.rule = rule
            self.value: Any = missing

        def __eq__(self, other: object) -> bool:
            return isinstance(other, MergeChannel) and other.rule.name == self.rule.name

        def __hash__(self) -> int:
            return hash(("aa-merge", self.rule.name))

        @property
        def ValueType(self) -> Any:
            return self.typ

        @property
        def UpdateType(self) -> Any:
            return self.typ

        def copy(self) -> Any:
            return self._with(self.value)

        def checkpoint(self) -> Any:
            return self.value

        def from_checkpoint(self, checkpoint: Any) -> Any:
            return self._with(checkpoint)

        def _with(self, value: Any) -> Any:
            channel = MergeChannel(self.rule, self.typ, self.key)
            channel.value = value
            return channel

        def get(self) -> Any:
            if self.value is missing:
                raise EmptyChannelError()
            return self.value

        def is_available(self) -> bool:
            return self.value is not missing

        def update(self, values: Sequence[Any]) -> bool:
            if not values:
                return False
            groups, plain = _groups(values, self.key)
            for index, group in enumerate(groups):
                if index == 0 and plain and self.value is missing and len(group) == 1:
                    self.value = group[0]  # the first input sets the field
                else:
                    self.value = self._merge(group)
            return True

        def _merge(self, group: list[Any]) -> Any:
            current = None if self.value is missing else self.value
            rule = self.rule
            try:
                return rule.merge(current, group)
            except FlowError as exc:
                raise FlowError(f"{exc} to {self.key}") from exc
            except (TypeError, ValueError) as exc:
                raise FlowError(
                    f"cannot merge {self.key} with rule {rule.name}: {type(exc).__name__}"
                ) from exc

    return MergeChannel
