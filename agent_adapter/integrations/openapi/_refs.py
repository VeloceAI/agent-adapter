"""Bounded `$ref` inlining and document size checks for untrusted OpenAPI documents."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote

from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

MAX_DEPTH = 128  # nesting levels, before and after inlining
MAX_DOCUMENT_NODES = 2_000_000  # values in the document with YAML aliases expanded
MAX_SCHEMA_NODES = 10_000  # values in one resolved schema, parameter or body
# Budgets relative to the document's size, so a small document cannot make an import do
# (or hold) much more work than it is big: values built while inlining (a repeated $ref is
# built once and shared), and values the imported tools hold (each tool copies its schemas).
INLINE_PER_VALUE = 64
MIN_INLINED = 50_000
MAX_INLINED = 1_000_000  # values built; half of it for response schemas
MAX_HELD = 4_000_000  # values held by all tools, expanded
# Top-level input fields of all tools: each one is a pydantic field the import builds.
FIELDS_PER_VALUE = 8
MIN_FIELDS = 5_000
MAX_FIELDS = 100_000


def budgets(values: int) -> tuple[int, int]:
    """(required, optional) build budgets for a document of `values` JSON values."""
    required = min(MAX_INLINED, max(MIN_INLINED, INLINE_PER_VALUE * values))
    return required, required // 2


def held_budget(values: int) -> int:
    """Expanded values all resolved schemas of a document of `values` JSON values may hold."""
    return min(MAX_HELD, max(MIN_INLINED, INLINE_PER_VALUE * values))


def fields_budget(values: int) -> int:
    """Top-level input fields all tools of a document of `values` JSON values may have."""
    return min(MAX_FIELDS, max(MIN_FIELDS, FIELDS_PER_VALUE * values))


class SchemaTooLarge(ConfigError):
    """A required result is over the per-result cap: its operation is skipped, not refused."""


class _Overflow(Exception):
    """Internal: an optional resolve went over a cap or budget; it is dropped, never raised."""


def check_document(document: Any) -> int:
    """The document's size in JSON values (aliases expanded); refuses too deep or too large.

    Raises:
        ConfigError: too deep, too large, or a value that contains itself.
    """
    sizes: dict[int, int] = {}  # id -> expanded size; aliases are counted, not walked again
    active: set[int] = set()

    def size(node: Any, depth: int) -> int:
        if depth > MAX_DEPTH:
            raise ConfigError(f"the OpenAPI document is too deep (over {MAX_DEPTH} levels)")
        if not isinstance(node, dict | list):
            return 1
        key = id(node)
        if key in sizes:
            return sizes[key]
        if key in active:
            raise ConfigError("the OpenAPI document contains a value that contains itself")
        active.add(key)
        children = node.values() if isinstance(node, dict) else node
        total = 1
        for child in children:
            total += size(child, depth + 1)
            if total > MAX_DOCUMENT_NODES:
                raise ConfigError(
                    f"the OpenAPI document is too large (over {MAX_DOCUMENT_NODES} values "
                    "with aliases expanded)"
                )
        active.discard(key)
        sizes[key] = total
        return total

    return size(document, 0)


class Resolver:
    """Inlines local `$ref`s of one document; each `resolve` result is capped in size.

    A repeated reference resolves to the same (shared) object, so the work is linear in
    the document. Three bounds: each result's expanded size (`limit`), the values built
    (`budgets`: a shared value is built once) and the expanded values all results hold
    (`held_budget`: every tool keeps its own copy). Results are never mutated: consumers
    build new containers (`normalise`, `copy.deepcopy`, renderers).
    """

    def __init__(
        self, document: Mapping[str, Any], *, limit: int = MAX_SCHEMA_NODES, values: int = 0
    ) -> None:
        self.document = document
        self.limit = limit
        # (ref, depth, None) for results without a cut recursion, else (ref, depth, path):
        # either way the value is fully determined by the key, so hits are exact.
        self._memo: dict[tuple[str, int, tuple[str, ...] | None], tuple[Any, int, bool]] = {}
        self._warned: set[str] = set()
        # Values consumers derive from shared results (`normalise`), by id; each entry keeps
        # its source alive so an id is never reused while the resolver lives.
        self.derived: dict[int, tuple[Any, Any]] = {}
        self._used = 0  # expanded values of the current result
        self._optional = False
        self._built = {False: 0, True: 0}  # values built so far: required, optional
        self._held = 0
        size = values or check_document(document)
        required, optional = budgets(size)
        self._budgets = {False: required, True: optional}
        self._held_budget = held_budget(size)

    def resolve(
        self, node: Any, *, optional: bool = False, deferred: list[int] | None = None
    ) -> Any:
        """`node` with its `$ref`s inlined; a recursive reference becomes `{}`.

        An `optional` resolve returns None instead of raising when it goes over a bound.
        With `deferred`, the result's size is appended there instead of being held: the
        caller holds it with `hold` once it keeps the result.

        Raises:
            SchemaTooLarge: a required result over the per-result cap (size or depth).
            ConfigError: a remote or broken reference, or a document-wide budget exceeded.
        """
        self._used, self._optional = 0, optional
        try:
            value, size, _ = self._walk(node, (), 0)
            if deferred is None:
                self._hold(size)
            else:
                deferred.append(size)
        except _Overflow:
            return None
        return value

    def hold(self, sizes: list[int]) -> None:
        """Hold the results resolved with `deferred=` (a kept operation's request parts).

        Raises:
            ConfigError: the document-wide held budget is exceeded.
        """
        self._optional = False
        self._hold(sum(sizes))

    def _hold(self, size: int) -> None:
        if self._held + size > self._held_budget:
            budget = self._held_budget
            self._overflow(
                f"the imported tools would hold more than their budget of {budget} schema "
                f"values ({INLINE_PER_VALUE} per document value, {MIN_INLINED} to {MAX_HELD}): "
                "the document is too large"
            )
        self._held += size

    def _charge(self, amount: int) -> None:
        """Count `amount` newly built values of the current result."""
        self._expand(amount)
        self._build(amount)

    def _build(self, amount: int) -> None:
        self._built[self._optional] += amount
        budget = self._budgets[self._optional]
        if self._built[self._optional] > budget:
            self._overflow(
                f"the whole document needs more than its inlining budget of {budget} values "
                f"({INLINE_PER_VALUE} per document value, {MIN_INLINED} to {MAX_INLINED}): "
                "it is too large"
            )

    def _expand(self, amount: int) -> None:
        """Count `amount` values (built or shared) against the current result's cap."""
        self._used += amount
        if self._used > self.limit:
            self._overflow(
                f"an OpenAPI schema is too large after inlining $refs (over {self.limit} values)",
                per_result=True,
            )

    def _overflow(self, message: str, *, per_result: bool = False) -> None:
        if self._optional:
            raise _Overflow(message)
        raise (SchemaTooLarge if per_result else ConfigError)(message)

    def _walk(self, node: Any, seen: tuple[str, ...], depth: int) -> tuple[Any, int, bool]:
        """(value, size, whether a recursive ref was cut below)."""
        if depth > MAX_DEPTH:
            self._overflow(
                f"an OpenAPI schema is too deep (over {MAX_DEPTH} levels)", per_result=True
            )
        if isinstance(node, list):
            return self._container(enumerate(node), seen, depth, as_list=True)
        if not isinstance(node, Mapping):
            self._charge(1)
            return node, 1, False
        ref = node.get("$ref")
        if not isinstance(ref, str):
            return self._container(node.items(), seen, depth, as_list=False)
        target, size, cut = self._ref(ref, seen, depth)
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        if siblings and isinstance(target, dict):  # OpenAPI 3.1: e.g. a description next to $ref
            extra, extra_size, extra_cut = self._container(
                siblings.items(), seen, depth, as_list=False
            )
            self._build(len(target))  # the merged copy is new; its values are counted
            return {**target, **extra}, size + extra_size, cut or extra_cut
        return target, size, cut

    def _container(
        self, items: Any, seen: tuple[str, ...], depth: int, *, as_list: bool
    ) -> tuple[Any, int, bool]:
        self._charge(1)
        out: dict[Any, Any] = {}
        total, cut = 1, False
        for key, child in items:
            value, size, child_cut = self._walk(child, seen, depth + 1)
            out[key] = value
            total, cut = total + size, cut or child_cut
        return (list(out.values()) if as_list else out), total, cut

    def _ref(self, ref: str, seen: tuple[str, ...], depth: int) -> tuple[Any, int, bool]:
        if ref in seen:
            if ref not in self._warned:
                self._warned.add(ref)
                logger.warning("recursive $ref %s replaced by an open schema", ref)
            self._charge(1)
            return {}, 1, True
        hit = self._memo.get((ref, depth, None)) or self._memo.get((ref, depth, seen))
        if hit is not None:
            value, size, cut = hit
            self._expand(size)  # shared, never copied: nothing new is built
            return value, size, cut
        value, size, cut = self._walk(pointer(self.document, ref), (*seen, ref), depth + 1)
        self._memo[(ref, depth, seen if cut else None)] = (value, size, cut)
        return value, size, cut


def resolve(node: Any, document: Mapping[str, Any]) -> Any:
    """`node` with every local `$ref` inlined (see `Resolver.resolve`)."""
    return Resolver(document).resolve(node)


def pointer(document: Mapping[str, Any], ref: str) -> Any:
    """The value a local JSON pointer `#/...` names (`~1`, `~0` and %-escapes decoded)."""
    if not ref.startswith("#/"):
        raise ConfigError(f"only local $refs are supported, got {ref!r}")
    node: Any = document
    for raw in ref[2:].split("/"):
        key = unquote(raw).replace("~1", "/").replace("~0", "~")
        if isinstance(node, Mapping) and key in node:
            node = node[key]
        elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        else:
            raise ConfigError(f"unresolvable $ref {ref!r}")
    return node
