"""Single-use resumes: which call keys may resume, claimed atomically, never twice."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, Literal, NoReturn

from agent_adapter.approval.store import ApprovalRecord, ApprovalStore
from agent_adapter.core.errors import ApprovalAlreadyDecided, ApprovalNotFound, ToolUsageError
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec.control import is_control_flow

logger = logging.getLogger(__name__)

Claim = Literal["absent", "mismatch", "claimed"]
STATE_TTL_S = 600.0  # the SDK's default requestState lifetime
PER_PRINCIPAL = 1_000  # pending resume keys kept per principal
MAX_PRINCIPALS = 10_000
MAX_SPENT = 100_000
PER_PRINCIPAL_SPENT = 1_000  # unexpired spent states kept per principal
INVALID_STATE = "Invalid or expired requestState"


def call_arguments(spec: ToolSpec, args: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validated arguments as the executor's approval sees them; None when invalid."""
    try:
        validated = spec.validate_args(args)
    except Exception as exc:  # the executor reports invalid arguments itself
        logger.debug(
            "resume check: arguments of %s are invalid (%s)", spec.name, type(exc).__name__
        )
        return None
    dumped: dict[str, Any] = validated.model_dump(mode="json", by_alias=True, fallback=str)
    return dumped


def arguments_digest(arguments: Mapping[str, Any] | None) -> str | None:
    if arguments is None:
        return None
    payload = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()


def refuse(message: str, code: str = "INVALID_PARAMS") -> NoReturn:
    """Raise the protocol error MCP clients see; `message` never holds ids or arguments."""
    import mcp.types as t
    from mcp import MCPError

    raise MCPError(code=getattr(t, code), message=message)


class ResumeKeys:
    """Resume keys of pending results when no shared store can vouch for them (per process).

    `claim` pops a key atomically, so two resumes never both run; a pending outcome
    issues it again. Capped per principal so one client cannot evict another's keys.
    """

    def __init__(
        self, *, per_principal: int = PER_PRINCIPAL, max_principals: int = MAX_PRINCIPALS
    ) -> None:
        self._keys: OrderedDict[str | None, OrderedDict[tuple[str, str, str], str | None]] = (
            OrderedDict()
        )
        self._per_principal = per_principal
        self._max_principals = max_principals
        self._lock = threading.Lock()

    def issue(
        self, principal: str | None, run_key: str, call_key: str, tool: str, digest: str | None
    ) -> None:
        with self._lock:
            keys = self._keys.setdefault(principal, OrderedDict())
            self._keys.move_to_end(principal)
            keys[(run_key, call_key, tool)] = digest
            keys.move_to_end((run_key, call_key, tool))
            if len(keys) > self._per_principal:
                keys.popitem(last=False)
                logger.warning(
                    "dropped a principal's oldest MCP resume key (cap %d)", self._per_principal
                )
            if len(self._keys) > self._max_principals:
                self._keys.popitem(last=False)
                logger.warning("dropped the MCP resume keys of the least recent principal")

    def claim(
        self, principal: str | None, run_key: str, call_key: str, tool: str, digest: str | None
    ) -> Claim:
        """`claimed` (and removed), `mismatch` (other arguments: kept) or `absent`."""
        with self._lock:
            keys = self._keys.get(principal)
            entry = (run_key, call_key, tool)
            if keys is None or entry not in keys:
                return "absent"
            if keys[entry] != digest:
                return "mismatch"
            del keys[entry]
            if not keys:
                del self._keys[principal]
            return "claimed"


class InFlight:
    """Calls being resumed in this process right now; one resume per call at a time."""

    def __init__(self) -> None:
        self._calls: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def claim(self, call: tuple[str, str]) -> bool:
        with self._lock:
            if call in self._calls:
                return False
            self._calls.add(call)
            return True

    def release(self, call: tuple[str, str]) -> None:
        with self._lock:
            self._calls.discard(call)


class SpentStates:
    """Client-approval states already used in this process, until their own deadline.

    The deadline is the state's wall-clock `iat` + TTL, read on the same `clock` that decides
    whether a state is still valid, so an entry never expires while its state is usable.
    Unexpired entries are never evicted: a principal over its quota, or a full set, refuses
    new states instead, so one principal cannot crowd out the others.
    """

    def __init__(
        self,
        *,
        max_entries: int = MAX_SPENT,
        per_principal: int = PER_PRINCIPAL_SPENT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._spent: dict[tuple[str, str], tuple[float, str | None]] = {}
        self._by_principal: dict[str | None, dict[tuple[str, str], float]] = {}
        self._max = max_entries
        self._per_principal = per_principal
        self.clock = clock
        self._lock = threading.Lock()

    def use(self, call: tuple[str, str], deadline: float, principal: str | None = None) -> bool:
        """True the first time `call` is used before `deadline` (test-and-set).

        `principal` picks the quota bucket; None and "" (anonymous) share one.

        Raises:
            MCPError: the principal's quota or the whole set is full of unexpired states
                (`INTERNAL_ERROR`).
        """
        bucket = principal or None
        now = self.clock()
        if deadline <= now:
            return False
        with self._lock:
            entry = self._spent.get(call)
            if entry is not None and entry[0] > now:
                return False
            if entry is not None:
                self._drop(call)
            if len(self._by_principal.get(bucket, ())) >= self._per_principal:
                self._sweep(now, self._by_principal[bucket])
            if len(self._by_principal.get(bucket, ())) >= self._per_principal:
                logger.warning(
                    "MCP approval states of one principal at quota (%d unexpired): refusing",
                    self._per_principal,
                )
                refuse("approval states busy", "INTERNAL_ERROR")
            if len(self._spent) >= self._max:
                self._sweep(now, {key: d for key, (d, _) in self._spent.items()})
            if len(self._spent) >= self._max:
                logger.warning("MCP approval-state set full (%d unexpired): refusing", self._max)
                refuse("approval states busy", "INTERNAL_ERROR")
            self._spent[call] = (deadline, bucket)
            self._by_principal.setdefault(bucket, {})[call] = deadline
            return True

    def _sweep(self, now: float, entries: Mapping[tuple[str, str], float]) -> None:
        for call in [call for call, deadline in entries.items() if deadline <= now]:
            self._drop(call)

    def _drop(self, call: tuple[str, str]) -> None:
        _, bucket = self._spent.pop(call)
        entries = self._by_principal[bucket]
        del entries[call]
        if not entries:
            del self._by_principal[bucket]


async def store_resumable(
    store: ApprovalStore, run_id: str, call_id: str, tool: str
) -> ApprovalRecord | None:
    """The call's unconsumed record for `tool` in the shared store, or None.

    Raises:
        MCPError: the store failed (fails closed).
    """
    try:
        record = await store.find(run_id, call_id)
    except Exception as exc:
        if is_control_flow(exc):
            raise
        logger.warning("approval store failed during an MCP resume check: %s", type(exc).__name__)
        refuse("approval store unavailable", "INTERNAL_ERROR")
    return record if record is not None and record.request.tool == tool else None


async def record_client_decision(
    store: ApprovalStore,
    spec: ToolSpec,
    arguments: dict[str, Any],
    run_id: str,
    call_id: str,
    decision: ApprovalDecision,
) -> None:
    """Create, decide and consume one record per approved state, so a replay on any worker
    sharing `store` is refused.

    Raises:
        MCPError: the state was already used, or the store failed.
    """
    seed = json.dumps([run_id, call_id], separators=(",", ":"), ensure_ascii=False)
    approval_id = "apr_" + hashlib.sha256(seed.encode("utf-8", "surrogatepass")).hexdigest()[:32]
    request = ApprovalRequest(
        approval_id=approval_id,
        run_id=run_id,
        call_id=call_id,
        tool=spec.name,
        arguments=arguments,
        hint="decided by the MCP client",
        created_at=time.time(),
    )
    try:
        record = await store.create(request)
        if record.approval_id != approval_id:
            _replayed(spec)
        await store.decide(approval_id, decision)
        if not await store.consume(approval_id):
            _replayed(spec)
    except (ToolUsageError, ApprovalAlreadyDecided, ApprovalNotFound):
        _replayed(spec)
    except Exception as exc:
        if is_control_flow(exc) or _is_mcp_error(exc):
            raise
        logger.warning(
            "approval store failed recording an MCP client decision: %s", type(exc).__name__
        )
        refuse("approval store unavailable", "INTERNAL_ERROR")


def _replayed(spec: ToolSpec) -> NoReturn:
    logger.warning("refusing a reused client-approval state for tool %s", spec.name)
    refuse(INVALID_STATE)


def _is_mcp_error(exc: BaseException) -> bool:
    from mcp import MCPError

    return isinstance(exc, MCPError)
