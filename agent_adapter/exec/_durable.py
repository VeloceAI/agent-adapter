"""Crash-safe runs, shared by flows and the tool loop: the call log, replay seeds, sealing.

Every journaled call of a durable run is logged twice in the run's current segment: as
`started` before it can run, and as `done` (its replayable result), `pending` or
`unrecordable` after. Recovery replays the `done` ones; a `started` or `unrecordable` one may
have run, so it re-runs only for an `idempotent=True` tool or when the caller allows it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from agent_adapter.core.errors import (
    CheckpointStoreCorrupt,
    ConfigError,
    RecoveryNeedsDecision,
    ToolUsageError,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ToolResult
from agent_adapter.exec._canon import canonical_bytes
from agent_adapter.exec._durable_seal import call_mac
from agent_adapter.exec._durable_seal import open_document as open_document
from agent_adapter.exec._durable_seal import seal_document as seal_document
from agent_adapter.exec._journal import JournalData, JournalSlot
from agent_adapter.exec.checkpoints import CallRecord, CheckpointStore, MemoryCheckpointStore
from agent_adapter.exec.checkpoints._port import IDEMPOTENCY_KEY
from agent_adapter.runtime.ids import new_id

logger = logging.getLogger(__name__)

_PROCESS_KEY = secrets.token_bytes(32)  # a memory store's runs only recover in this process


def durable_key(store: object, key: bytes | None, owner: str) -> bytes:
    """The key durable checkpoints are signed with.

    Raises:
        ToolUsageError: `store` is not a CheckpointStore, or `key` is not 16+ bytes.
        ConfigError: a store other than `MemoryCheckpointStore` without a key: its runs
            must recover in another process, which needs the same key.
    """
    if not isinstance(store, CheckpointStore):
        raise ToolUsageError(f"checkpoints must be a CheckpointStore, got {type(store).__name__}")
    if key is not None:
        if not isinstance(key, bytes) or len(key) < 16:
            raise ToolUsageError("checkpoint_key must be bytes of at least 16 bytes")
        return key
    if isinstance(store, MemoryCheckpointStore):
        return _PROCESS_KEY
    raise ConfigError(
        f"{owner}: a {type(store).__name__} needs checkpoint_key=, so another process can "
        "check and recover the runs it saves"
    )


def new_segment() -> str:
    """A fresh call-log segment id."""
    return new_id("seg_")


class CallLog:
    """The `JournalObserver` of one durable run: logs signed calls, hands out replay seeds."""

    def __init__(self, store: CheckpointStore, run_id: str, segment: str, key: bytes) -> None:
        self.store = store
        self.run_id = run_id
        self.segment = segment
        self.key = key
        self._seeds: dict[str, JournalData] = {}
        self._identities: dict[int, list[Any]] = {}  # started slot -> its identity
        self._unkeyed: dict[str, int] = {}
        self._seq: dict[str, int] = {}  # call key -> its sequence number in this segment
        self._expected: set[str] = set()  # keys of recovered `done` calls not replayed yet
        self.broken = False  # a write failed: the run stays recoverable, never marked failed

    def move_to(self, segment: str, seeds: Iterable[CallRecord] = ()) -> int:
        """Log to `segment` from now on; `seeds` (recovered calls) replay in the next step.

        Returns the number of recovered calls the step before never replayed (their
        arguments came out different, so those tools ran again).
        """
        missed = len(self._expected)
        if missed:
            logger.warning(
                "run %s: %d recovered call(s) did not replay (arguments changed?)",
                self.run_id,
                missed,
            )
        calls = list(seeds)
        self.segment = segment
        self._seq = {}
        self._seeds = seeds_from(calls)
        self._expected = {c.key for c in calls if c.state == "done" and c.result is not None}
        return missed

    def seed(self, root: str) -> JournalData | None:
        """The recovered calls of one root journal (a node visit or a loop turn), once."""
        return self._seeds.pop(root, None)

    async def replayed(self, path: tuple[Any, ...], slot: JournalSlot) -> None:
        """A recovered call replayed: it is no longer expected."""
        self._expected.discard(_key_of(self._identity(path, slot)))

    async def started(self, path: tuple[Any, ...], slot: JournalSlot) -> None:
        """Log that a call may run from now on; refused once another worker took the run."""
        identity = self._identity(path, slot)
        self._identities[id(slot)] = identity
        await self._write(identity, slot.tool, "started")
        IDEMPOTENCY_KEY.set(
            hashlib.sha256(f"{self.run_id}\0{_key_of(identity)}".encode()).hexdigest()
        )

    async def settled(
        self,
        path: tuple[Any, ...],
        slot: JournalSlot,
        result: ToolResult,
        stored: Mapping[str, Any] | None,
    ) -> None:
        """Log how a call ended: its replayable result, pending, or ran but unrecordable."""
        IDEMPOTENCY_KEY.set(None)
        identity = self._identities.pop(id(slot), None) or self._identity(path, slot)
        if result.status == "pending":
            await self._write(identity, slot.tool, "pending")
        elif stored is None or "unsealable" in stored:
            why = str((stored or {}).get("why", "")) or slot.why or "result not recorded"
            await self._write(identity, slot.tool, "unrecordable", why=why)
        else:
            await self._write(identity, slot.tool, "done", result=dict(stored))

    def _identity(self, path: tuple[Any, ...], slot: JournalSlot) -> list[Any]:
        root, children = str(path[0]) if path else "", [list(p) for p in path[1:]]
        occurrence: Any = slot.occurrence
        if not slot.digest:  # no canonical arguments: counted apart, never replayed
            where = json.dumps([root, children])
            occurrence = f"u{self._unkeyed.get(where, 0)}"
            self._unkeyed[where] = self._unkeyed.get(where, 0) + 1
        return [root, children, slot.tool, slot.digest, occurrence]

    async def _write(
        self,
        identity: list[Any],
        tool: str,
        state: Any,
        *,
        result: dict[str, Any] | None = None,
        why: str = "",
    ) -> None:
        key = _key_of(identity)
        seq = self._seq.setdefault(key, len(self._seq))
        record = CallRecord(
            key=key, tool=tool, state=state, result=result, why=why, identity=identity, seq=seq
        )
        record = record.model_copy(
            update={"mac": call_mac(self.key, self.run_id, self.segment, record)}
        )
        try:
            await self.store.record_call(self.run_id, self.segment, record)
        except BaseException:
            self.broken = True
            raise


def _key_of(identity: list[Any]) -> str:
    return hashlib.sha256(canonical_bytes(identity)).hexdigest()


def seeds_from(calls: Iterable[CallRecord]) -> dict[str, JournalData]:
    """Journal data per root from the `done` records of a segment."""
    trees: dict[str, JournalData] = {}
    for call in calls:
        if call.state != "done" or call.result is None:
            continue
        root, children, tool, digest, occurrence = _parts(call)
        node = trees.setdefault(root, {})
        for name, n in children:
            kids = node.setdefault("children", [])
            found = next((kid for kid in kids if kid[0] == name and kid[1] == n), None)
            if found is None:
                found = [name, n, {}]
                kids.append(found)
            node = found[2]
        node.setdefault("calls", []).append([tool, digest, occurrence, call.result])
    return trees


def merged(first: JournalData | None, extra: JournalData | None) -> JournalData | None:
    """Two journal trees as one; `first` wins where both hold the same call."""
    if not extra:
        return first
    if not first:
        return extra
    calls = {(t, d, n): r for t, d, n, r in extra.get("calls", ())}
    calls.update({(t, d, n): r for t, d, n, r in first.get("calls", ())})
    kids: dict[tuple[str, int], JournalData] = {
        (name, n): data for name, n, data in extra.get("children", ())
    }
    for name, n, data in first.get("children", ()):
        kids[(name, n)] = merged(data, kids.get((name, n))) or {}
    out: JournalData = {}
    if calls:
        out["calls"] = [[t, d, n, r] for (t, d, n), r in sorted(calls.items())]
    if kids:
        out["children"] = [[name, n, data] for (name, n), data in sorted(kids.items())]
    return out


def check_in_doubt(
    run_id: str,
    calls: Sequence[CallRecord],
    idempotent: Callable[[str], bool],
    *,
    rerun: bool,
) -> list[CallRecord]:
    """The calls that may have run but cannot replay; refuse unless each may run again.

    Raises:
        RecoveryNeedsDecision: such a call's tool is not `idempotent=True` and `rerun` is False.
    """
    doubtful = [c for c in calls if c.state in ("started", "unrecordable")]
    blocked = [c for c in doubtful if not idempotent(c.tool)]
    if blocked and not rerun:
        logger.warning(
            "run %s: %d call(s) in doubt after a crash; recovery needs a decision",
            run_id,
            len(blocked),
        )
        raise RecoveryNeedsDecision(
            run_id, [(c.tool, c.why or "started, no result logged") for c in blocked]
        )
    if doubtful:
        logger.warning("run %s: %d call(s) in doubt run again", run_id, len(doubtful))
    return doubtful


def idempotent_in(registry: Registry | None, extra: Mapping[str, bool] | None = None) -> Any:
    """Whether a tool name is marked `idempotent=True` (unknown tools are not)."""
    known = dict(extra or {})

    def check(tool: str) -> bool:
        if tool in known:
            return known[tool]
        if registry is not None and tool in registry:
            return registry.get(tool).idempotent is True
        return False

    return check


def _parts(call: CallRecord) -> tuple[str, list[tuple[str, int]], str, str, Any]:
    """The identity of a logged call.

    Raises:
        CheckpointStoreCorrupt: the identity is not the shape `CallLog` writes.
    """
    try:
        root, children, tool, digest, occurrence = call.identity
        path = [(str(name), int(n)) for name, n in children]
        if not isinstance(root, str) or not isinstance(digest, str) or tool != call.tool:
            raise TypeError("identity fields")
        if call.key != hashlib.sha256(canonical_bytes(call.identity)).hexdigest():
            raise ValueError("the key is not the identity's digest")
        return root, path, tool, digest, occurrence
    except (TypeError, ValueError) as exc:
        logger.error("a logged call of tool %s has an unreadable identity", call.tool)
        raise CheckpointStoreCorrupt("a logged call has an unreadable identity") from exc
