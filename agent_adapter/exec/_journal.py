"""The replay journal of one flow node visit: finished executor calls, as sealed JSON data.

Only the flow runner makes journals (`_make_journal`). A journal holds plain JSON: each result
in `ToolResult.to_dict()` form with its value in the tagged canonical encoding, and an error's
exception rebuilt from `ErrorInfo` on replay, so a checkpoint never carries live objects.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import threading
from collections.abc import Collection, Iterator, Mapping
from typing import Any

from agent_adapter.core.errors import AdapterError, ToolUsageError, log_failure, qualified_name
from agent_adapter.core.types import ErrorInfo, ToolResult
from agent_adapter.exec._canon import (
    Decoder,
    Encoder,
    NotSealable,
    decode,
    digest,
    encode,
    reachable_types,
    tagged_classes,
)

logger = logging.getLogger(__name__)

_TOKEN = object()  # module-private: only `_make_journal` constructs a Journal

JournalData = dict[str, Any]  # {"calls": [[tool, digest, n, result]], "children": [[key, n, data]]}

RAN_UNSEALABLE = "ran, result not sealable"
RAN_UNKEYED = "ran, arguments not sealable"


@dataclasses.dataclass(frozen=True)
class JournalSlot:
    """One executor call of a node visit: its key, and the result to replay (if recorded)."""

    tool: str
    digest: str  # "" when the arguments have no canonical form (the call cannot replay)
    occurrence: int
    result: ToolResult | None
    allowed: Mapping[str, type] = dataclasses.field(default_factory=dict)  # replayable classes
    why: str = ""  # why the call has no key


class Journal:
    """The finished executor calls of one node visit, replayed when a resumed node re-runs.

    A call is keyed by tool, the canonical digest of its validated arguments, and how many
    calls with that key came before it in the visit. Child journals (a `Map` item, a nested
    flow's node) are keyed by name and occurrence. Pending results are never recorded; a call
    that ran but cannot be recorded leaves a marker, so a pause fails instead of re-running it.
    """

    def __init__(
        self,
        token: object,
        data: JournalData | None = None,
        *,
        encoder: Encoder | None = None,
        decoder: Decoder | None = None,
        calls: bool = True,
        allowed: Mapping[str, type] | None = None,
    ) -> None:
        if token is not _TOKEN:
            raise ToolUsageError("Journal is made by the flow runner only")
        data = data or {}
        self._results: dict[tuple[str, str, int], dict[str, Any]] = {
            (tool, key, n): result for tool, key, n, result in data.get("calls", ())
        }
        self._seeds: dict[tuple[str, int], JournalData] = {
            (name, n): child for name, n, child in data.get("children", ())
        }
        self._children: dict[tuple[str, int], Journal] = {}
        self._seen: dict[tuple[str, str], int] = {}
        self._child_seen: dict[str, int] = {}
        # a custom value is only recorded when it can also be decoded for the replay
        self._encoder = encoder if decoder is not None else None
        self._decoder = decoder
        self._calls = calls
        self._allowed = dict(allowed or {})  # classes a replayed value may hold
        self._unkeyed = 0
        self._lock = threading.Lock()
        self._closed = False

    def claim(self, tool: str, arguments: Any, returns: Any = None) -> JournalSlot | None:
        """The next slot for this call (None when it is not journaled).

        Args:
            returns: The tool's return annotation; the classes it reaches may be replayed.

        Raises:
            ToolUsageError: the call already ran in this visit with a result that could not
                be recorded (it must not run again).
        """
        if not self._calls:
            return None
        allowed = {**self._allowed, **_return_types(returns)}
        try:
            key = digest({"tool": tool, "arguments": arguments}, self._encoder)
        except NotSealable as exc:
            return JournalSlot(tool, "", 0, None, allowed, why=str(exc))
        with self._lock:
            if self._closed:
                return None
            occurrence = self._seen.get((tool, key), 0)
            self._seen[(tool, key)] = occurrence + 1
            stored = self._results.get((tool, key, occurrence))
        if stored is not None and "unsealable" in stored:
            raise ToolUsageError(
                f"tool {tool!r} already ran in this node visit and its result could not be "
                "recorded; it does not run again"
            )
        return JournalSlot(tool, key, occurrence, self._replayed(stored, allowed), allowed)

    def record(self, slot: JournalSlot, result: ToolResult) -> None:
        """Keep a finished result for replay, or a marker when it cannot be kept."""
        if result.status == "pending":
            return
        if slot.why:
            self._keep(slot, {"unsealable": RAN_UNKEYED, "why": slot.why}, unkeyed=True)
            return
        try:
            data = _dump_result(result, self._encoder)
            if self._decoder is None:
                _check_classes(data, slot.allowed)
        except NotSealable as exc:
            log_failure(
                logger, logging.WARNING, exc, "tool %s: result cannot be journaled", slot.tool
            )
            data = {"unsealable": RAN_UNSEALABLE, "why": str(exc), "call_id": result.call_id}
        self._keep(slot, data)

    def _keep(self, slot: JournalSlot, data: dict[str, Any], *, unkeyed: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            key = (slot.tool, slot.digest, slot.occurrence)
            if unkeyed:
                key = (slot.tool, "", self._unkeyed)
                self._unkeyed += 1
            self._results.setdefault(key, data)

    def child(self, name: str, seed: JournalData | None = None, *, calls: bool = True) -> Journal:
        """The next child journal named `name` (recorded one, else a new one from `seed`)."""
        with self._lock:
            n = self._child_seen.get(name, 0)
            self._child_seen[name] = n + 1
            data = self._seeds.pop((name, n), seed)
            child = Journal(
                _TOKEN,
                data,
                encoder=self._encoder,
                decoder=self._decoder,
                calls=calls,
                allowed=self._allowed,
            )
            if self._closed:
                child.close()
            self._children[(name, n)] = child
            return child

    def data(self, drop: Collection[str] = ()) -> JournalData:
        """Everything recorded, as JSON data; children named in `drop` are left out."""
        with self._lock:
            calls = [[t, d, n, r] for (t, d, n), r in sorted(self._results.items())]
            kids = {**self._seeds, **{k: c.data() for k, c in self._children.items()}}
        children = [[k, n, d] for (k, n), d in sorted(kids.items()) if k not in drop and d]
        out: JournalData = {}
        if calls:
            out["calls"] = calls
        if children:
            out["children"] = children
        return out

    def close(self) -> None:
        """Record and replay nothing from now on (children too)."""
        with self._lock:
            self._closed = True
            children = list(self._children.values())
        for child in children:
            child.close()

    def _replayed(
        self, stored: dict[str, Any] | None, allowed: Mapping[str, type]
    ) -> ToolResult | None:
        if stored is None:
            return None
        try:
            return _load_result(stored, self._decoder, allowed)
        except (NotSealable, ValueError, TypeError, KeyError) as exc:
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "journaled result cannot be replayed; the call runs again",
            )
            return None


def _make_journal(
    data: JournalData | None = None,
    *,
    encoder: Encoder | None = None,
    decoder: Decoder | None = None,
    calls: bool = True,
    allowed: Mapping[str, type] | None = None,
) -> Journal:
    return Journal(_TOKEN, data, encoder=encoder, decoder=decoder, calls=calls, allowed=allowed)


def unsealable_calls(data: JournalData, where: str = "") -> list[str]:
    """Calls in journal data that ran but could not be recorded (children included)."""
    found = [
        f"call {tool!r}{where} (occurrence {n}): {result['unsealable']} ({result.get('why', '')})"
        for tool, _key, n, result in data.get("calls", ())
        if "unsealable" in result
    ]
    for name, n, child in data.get("children", ()):
        found.extend(unsealable_calls(child, f"{where} in {name}#{n}"))
    return found


@functools.lru_cache(maxsize=1024)
def _cached_types(returns: Any) -> Mapping[str, type]:
    return reachable_types(returns)


def _return_types(returns: Any) -> Mapping[str, type]:
    if returns is None:
        return {}
    try:
        return _cached_types(returns)
    except TypeError:  # an unhashable annotation (e.g. Annotated metadata)
        return reachable_types(returns)


def _check_classes(data: dict[str, Any], allowed: Mapping[str, type]) -> None:
    """Refuse a value holding a class its replay would refuse."""
    names = tagged_classes(data["value"]) | tagged_classes(data.get("exception_state"))
    missing = sorted(names - allowed.keys())
    if missing:
        raise NotSealable(
            "$.value",
            missing[0],
            "class not reachable from the tool's return annotation or the state model",
        )


def _dump_result(result: ToolResult, encoder: Encoder | None) -> dict[str, Any]:
    data = result.to_dict()
    data["value"] = encode(result.value, encoder, "$.value")
    exc = result.exception
    if exc is not None:
        if not isinstance(exc, AdapterError):
            raise NotSealable("$.exception", type(exc).__qualname__, "not an AdapterError")
        state = {k: v for k, v in exc.__dict__.items() if not k.startswith("_")}
        data["exception_state"] = encode(state, None, "$.exception")
    return data


def _load_result(
    data: dict[str, Any], decoder: Decoder | None, allowed: Mapping[str, type]
) -> ToolResult:
    error = ErrorInfo.model_validate(data["error"]) if data["error"] is not None else None
    exception = None
    if error is not None:
        state = decode(data.get("exception_state", {}), decoder, allowed)
        exception = _rebuilt(error, state)
    return ToolResult(
        call_id=str(data["call_id"]),
        name=str(data["name"]),
        status=data["status"],
        value=decode(data["value"], decoder, allowed),
        error=error,
        exception=exception,
        attempts=int(data["attempts"]),
        cached=bool(data["cached"]),
        latency_ms=float(data["latency_ms"]),
    )


def _rebuilt(error: ErrorInfo, state: dict[str, Any]) -> AdapterError:
    """The error's exception: its AdapterError class (already loaded), message and attributes."""
    cls = next((c for c in _adapter_errors() if qualified_name(c) == error.exception), None)
    if cls is None:
        raise NotSealable("$.exception", error.exception, "not a known AdapterError")
    exc = cls.__new__(cls)
    exc.args = (error.message,)
    exc.__dict__.update(state)
    return exc


def _adapter_errors() -> Iterator[type[AdapterError]]:
    todo: list[type[AdapterError]] = [AdapterError]
    while todo:
        cls = todo.pop()
        yield cls
        todo.extend(cls.__subclasses__())
