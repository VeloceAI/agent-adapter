"""Signing for durable runs: the checkpoint envelope, each logged call, and the run's identity.

The store is untrusted. Every field recovery acts on is under an HMAC with the checkpoint key:
the envelope binds run id, step, kind, name, status, segment, the run's identity, the calls
carried over from a crashed step and a pending approval spend; each logged call binds run id,
segment, its key, state, result, identity and its sequence number in the segment.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import CheckpointStoreCorrupt, ToolUsageError
from agent_adapter.exec._canon import canonical_bytes
from agent_adapter.exec.checkpoints import CallRecord, Checkpoint, StoredCheckpoint

logger = logging.getLogger(__name__)

ENVELOPE = "agent_adapter.durable.checkpoint/1"
IDENTITY_FIELDS = ("user_id", "thread_id", "principal")


@dataclass(frozen=True)
class Envelope:
    """A stored checkpoint, opened and checked against its row."""

    body: Any  # a flow's sealed checkpoint text, or a tool loop's payload
    identity: Mapping[str, str]
    carried: tuple[CallRecord, ...]  # calls of the crashed step, carried into this segment
    spend: Mapping[str, Any] | None  # an approval decision stored but maybe not yet spent


def seal_document(key: bytes, form: str, payload: Mapping[str, Any]) -> str:
    """A signed JSON document: format, payload and HMAC-SHA256 of the canonical payload."""
    mac = hmac.new(key, canonical_bytes(payload), hashlib.sha256).hexdigest()
    return json.dumps({"format": form, "payload": payload, "mac": mac}, sort_keys=True)


def open_document(key: bytes, form: str, text: str, run_id: str) -> dict[str, Any]:
    """The payload of a signed document, after its format and MAC are checked.

    Raises:
        CheckpointStoreCorrupt: not JSON, another format, or a MAC that does not match.
    """
    try:
        document = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise corrupt(run_id, "not JSON") from exc
    if not isinstance(document, dict) or document.get("format") != form:
        raise corrupt(run_id, "not a checkpoint document of this kind")
    payload, given = document.get("payload"), document.get("mac")
    if not isinstance(payload, dict) or not isinstance(given, str):
        raise corrupt(run_id, "payload or MAC missing")
    expected = hmac.new(key, canonical_bytes(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(given, expected):
        raise corrupt(run_id, "the MAC does not match (another key, or changed)")
    return payload


def seal_checkpoint(
    key: bytes,
    run_id: str,
    step: int,
    checkpoint: Checkpoint,
    body: Any,
    identity: Mapping[str, str],
    *,
    carried: Sequence[CallRecord] = (),
    spend: Mapping[str, Any] | None = None,
) -> Checkpoint:
    """`checkpoint` with its `data` set to the signed envelope for this run and step."""
    payload = {
        "run_id": run_id,
        "step": step,
        "kind": checkpoint.kind,
        "name": checkpoint.name,
        "status": checkpoint.status,
        "segment": checkpoint.segment,
        "identity": dict(identity),
        "carried": [call.model_dump(mode="json", exclude={"mac"}) for call in carried],
        "spend": dict(spend) if spend is not None else None,
        "body": body,
    }
    return checkpoint.model_copy(update={"data": seal_document(key, ENVELOPE, payload)})


def open_checkpoint(key: bytes, stored: StoredCheckpoint) -> Envelope:
    """The envelope of a stored checkpoint; every row field must match the signed one.

    Raises:
        CheckpointStoreCorrupt: a bad signature, or a row that disagrees with its envelope.
    """
    run_id, saved = stored.run_id, stored.checkpoint
    payload = open_document(key, ENVELOPE, saved.data, run_id)
    row = (run_id, stored.step, saved.kind, saved.name, saved.status, saved.segment)
    signed = tuple(payload.get(f) for f in ("run_id", "step", "kind", "name", "status", "segment"))
    if row != signed:
        raise corrupt(run_id, "the row disagrees with its signed checkpoint")
    try:
        identity = {f: str(payload["identity"].get(f, "")) for f in IDENTITY_FIELDS}
        carried = tuple(CallRecord.model_validate(c) for c in payload["carried"])
        spend = payload["spend"]
        if spend is not None and not isinstance(spend, dict):
            raise TypeError("spend is not an object")
    except (KeyError, TypeError, AttributeError, ValidationError) as exc:
        raise corrupt(run_id, "the signed checkpoint has an unreadable part") from exc
    return Envelope(payload.get("body"), identity, carried, spend)


def call_mac(key: bytes, run_id: str, segment: str, call: CallRecord) -> str:
    """The HMAC of one logged call, bound to its run and segment."""
    fields = call.model_dump(mode="json", exclude={"mac"})
    data = canonical_bytes({"run_id": run_id, "segment": segment, "call": fields})
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def verified_calls(
    key: bytes, run_id: str, segment: str, calls: Iterable[CallRecord]
) -> list[CallRecord]:
    """The logged calls of a segment, each signature checked and none missing in between.

    Raises:
        CheckpointStoreCorrupt: a call is unsigned or changed, or a sequence number is missing.
    """
    found = list(calls)
    for call in found:
        if not hmac.compare_digest(call.mac, call_mac(key, run_id, segment, call)):
            raise corrupt(run_id, "a logged call is not signed with this key or was changed")
    if sorted(call.seq for call in found) != list(range(len(found))):
        raise corrupt(run_id, "a logged call is missing from the store")
    return found


def merge_calls(carried: Sequence[CallRecord], current: Sequence[CallRecord]) -> list[CallRecord]:
    """The calls of a step across its segments: a later record of the same call wins."""
    out: dict[str, CallRecord] = {call.key: call for call in carried}
    for call in current:
        out[call.key] = call
    return list(out.values())


def identity_of(ctx: ToolContext) -> dict[str, str]:
    """The run's identity a checkpoint keeps: the end user, the thread and the principal."""
    return {"user_id": ctx.user_id, "thread_id": ctx.thread_id, "principal": principal_id(ctx)}


def principal_id(ctx: ToolContext) -> str:
    """A stable id of `ctx.extras["principal"]` (its `id`, or the text itself); "" when none."""
    principal = ctx.extras.get("principal")
    if principal is None:
        return ""
    if isinstance(principal, str):
        return principal
    found = (
        principal.get("id") if isinstance(principal, Mapping) else getattr(principal, "id", None)
    )
    return str(found) if found is not None else f"<{type(principal).__name__}>"


def check_identity(run_id: str, sealed: Mapping[str, str], ctx: ToolContext | None) -> None:
    """Refuse a context for another user, thread or principal than the run was saved with.

    Raises:
        ToolUsageError: `ctx` names another identity, or none is given for a run whose
            principal cannot be rebuilt from the checkpoint.
    """
    if ctx is None:
        if sealed.get("principal"):
            raise ToolUsageError(
                f"run {run_id!r} ran for a principal: recover it with that principal's ctx"
            )
        return
    given = identity_of(ctx)
    for name in IDENTITY_FIELDS:
        if given[name] != sealed.get(name, ""):
            logger.warning("run %s: recovery refused for another %s", run_id, name)
            raise ToolUsageError(f"run {run_id!r} belongs to another {name}; recovery refused")


def corrupt(run_id: str, why: str) -> CheckpointStoreCorrupt:
    """The error for store content that does not check out."""
    logger.error("run %s: stored checkpoint refused (%s)", run_id, why)
    return CheckpointStoreCorrupt(f"run {run_id!r}: stored checkpoint refused: {why}")
