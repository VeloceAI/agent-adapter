"""DeferredApproval: park a call in an ApprovalStore and resume it once someone decides."""

from __future__ import annotations

import logging
import os

from agent_adapter.approval.sqlite_store import SQLiteApprovalStore
from agent_adapter.approval.store import ApprovalRecord, ApprovalStore, request_digest
from agent_adapter.core.errors import (
    ApprovalDenied,
    ApprovalNotFound,
    ApprovalPending,
    ToolUsageError,
)
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.runtime.ids import new_id

logger = logging.getLogger(__name__)

CONCURRENT_USE = "approval already used by a concurrent request"
DECISION_CHANGED = "decision changed since it was prepared"
CALL_ID_REUSED = "call id reused with different tool/arguments"
# Metadata the HTTP server writes on the records it parks (`server/_approval.py`). It
# re-authorizes the invoker before it runs one, so only its own port may consume them.
SERVER_METADATA = ("origin", "principal", "anonymous")


def server_owned(record: ApprovalRecord) -> bool:
    """True when the HTTP tool server parked `record` (it alone may run it)."""
    return any(key in record.request.metadata for key in SERVER_METADATA)


class DeferredApproval:
    """ApprovalPort for out-of-band review (HTTP server, queues); emits no events itself."""

    owns_server_records = False  # only the HTTP server's own port sets this

    def __init__(self, store: ApprovalStore) -> None:
        self.store = store

    @classmethod
    def sqlite(
        cls, path: str | os.PathLike[str], *, max_records: int = 10_000, timeout: float = 5.0
    ) -> DeferredApproval:
        """A port on a `SQLiteApprovalStore` file, shared by every process on this host.

        Raises:
            ConfigError: a bad path, `max_records` or `timeout` (see `SQLiteApprovalStore`).
        """
        return cls(SQLiteApprovalStore(path, max_records=max_records, timeout=timeout))

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        """Return the decision for `(run_id, call_id)` once decided, consuming it.

        A record only answers the request it was made for: same kind, tool and original
        arguments (`args_digest`). The decision's `edited_args` are what then runs.

        Raises:
            ApprovalPending: no decision yet; the first request creates the record.
            ApprovalDenied: a concurrent request consumed the same decision (fails closed).
            ToolUsageError: the call id has a record for another tool or other arguments.
        """
        return await self.consume(await self.prepare(request), request)

    async def prepare(self, request: ApprovalRequest) -> ApprovalRecord:
        """The record for `(run_id, call_id)`, created pending when new; never consumes it.

        Advisory: batch callers (`approval.batch.ask_each`) prepare every call first and act
        only on what `consume` returns for these records.

        Raises:
            ToolUsageError: non-JSON arguments, the call id has a record for another tool
                or other arguments, or the HTTP tool server parked the record.
        """
        digest = request_digest(request)  # fails closed on non-JSON arguments, before any I/O
        record = await self.store.find(request.run_id, request.call_id)
        if record is None:
            # Atomic per call: a racer's open record comes back instead of a new one.
            record = await self.store.create(await self._fresh(request))
            if record.status == "pending":
                logger.info(
                    "approval %s pending for tool %r (run %s, call %s)",
                    record.approval_id,
                    request.tool,
                    request.run_id,
                    request.call_id,
                )
        self._check_same_call(record, request, digest)
        self._check_owner(record)
        return record

    def _check_owner(self, record: ApprovalRecord) -> None:
        if self.owns_server_records or not server_owned(record):
            return
        stored = record.request
        logger.warning(
            "approval %s (tool %r, run %s, call %s) was parked by the HTTP tool server: "
            "refusing to consume it outside the server",
            record.approval_id,
            stored.tool,
            stored.run_id,
            stored.call_id,
        )
        raise ToolUsageError(
            f"approval {record.approval_id!r} was parked by the HTTP tool server, which "
            f"re-authorizes the invoker: decide it there (POST /approvals/{record.approval_id}"
            "/decide)"
        )

    @staticmethod
    def _check_same_call(record: ApprovalRecord, request: ApprovalRequest, digest: str) -> None:
        # Resuming another call's approval would run new arguments under an old decision, or
        # spend another run's decision (the digest has no ids). Compare with the stored
        # request itself, never a stored digest.
        stored = record.request
        same_ids = (stored.run_id, stored.call_id) == (request.run_id, request.call_id)
        if same_ids and request_digest(stored) == digest:
            return
        logger.warning(
            "approval %s (tool %r, run %s, call %s) does not match the request for tool %r"
            " (run %s, call %s): refusing to reuse it",
            record.approval_id,
            stored.tool,
            stored.run_id,
            stored.call_id,
            request.tool,
            request.run_id,
            request.call_id,
        )
        raise ToolUsageError(f"{CALL_ID_REUSED} (tool {request.tool!r}, call {request.call_id!r})")

    async def consume(self, record: ApprovalRecord, request: ApprovalRequest) -> ApprovalDecision:
        """Use the stored decision of `record` once; the record and request are not trusted.

        The record is read again from the store; never looks the call up by `call_id`.

        Raises:
            ApprovalPending: the stored record is not decided (nothing is consumed).
            ApprovalDenied: the stored decision differs from `record`'s, or the record was
                already consumed, e.g. by a concurrent resume (fails closed).
            ToolUsageError: `request` is for another run, call, tool or arguments, or the
                HTTP tool server parked the record (nothing is consumed).
            ApprovalNotFound: the store has no such record.
        """
        current = await self.store.get(record.approval_id)
        self._check_same_call(current, request, request_digest(request))
        self._check_owner(current)
        if current.status == "pending" or current.decision is None:
            raise ApprovalPending(
                current.approval_id,
                tool=request.tool,
                run_id=request.run_id,
                call_id=request.call_id,
            )
        if current.decision != record.decision:
            logger.warning(
                "approval %s for tool %r (run %s, call %s): decision changed since it was "
                "prepared; denying",
                current.approval_id,
                request.tool,
                request.run_id,
                request.call_id,
            )
            raise ApprovalDenied(request.tool, call_id=request.call_id, reason=DECISION_CHANGED)
        if not await self.store.consume(current.approval_id):
            logger.warning(
                "approval %s for tool %r (run %s, call %s) was already consumed: denying",
                record.approval_id,
                request.tool,
                request.run_id,
                request.call_id,
            )
            raise ApprovalDenied(request.tool, call_id=request.call_id, reason=CONCURRENT_USE)
        return current.decision

    async def _fresh(self, request: ApprovalRequest) -> ApprovalRequest:
        # A resumed call may reuse the id of a record it already consumed.
        if not request.approval_id:
            return request
        try:
            await self.store.get(request.approval_id)
        except ApprovalNotFound:
            return request
        return request.model_copy(update={"approval_id": new_id("apr_")})
