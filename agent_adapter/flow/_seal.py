"""Sealing checkpoints: the canonical payload, its HMAC, the graph digest, and the JSON form."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.errors import FlowError, ToolUsageError, log_failure
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec._canon import (
    Decoder,
    Encoder,
    NotSealable,
    canonical_bytes,
    check_plain,
    decode,
    encode,
    reachable_types,
    tagged_classes,
)
from agent_adapter.flow._digest import graph_digest
from agent_adapter.flow._result import FlowCheckpoint, frozen
from agent_adapter.flow.graph import FlowGraph

logger = logging.getLogger(__name__)

TAMPERED = "checkpoint-tampered"
NOT_SEALABLE = "state-not-sealable"
FORMAT = "agent_adapter.flow.checkpoint/1"

# Without a configured key a checkpoint only resumes in the process that paused it.
_PROCESS_KEY = secrets.token_bytes(32)


@dataclass(frozen=True)
class Sealer:
    """A runner's key and its codec hooks for values the canonical encoding does not know."""

    key: bytes
    encoder: Encoder | None = None
    decoder: Decoder | None = None


def runner_key(key: bytes | None) -> bytes:
    """The checkpoint key a runner signs with.

    Raises:
        ToolUsageError: `key` is not bytes of at least 16 bytes.
    """
    if key is None:
        return _PROCESS_KEY
    if not isinstance(key, bytes) or len(key) < 16:
        raise ToolUsageError("checkpoint_key must be bytes of at least 16 bytes")
    return key


def allowed_classes(state_model: type[BaseModel]) -> dict[str, type]:
    """The classes a checkpoint of this state model may hold (`exec._canon.reachable_types`)."""
    return reachable_types(state_model, ApprovalRequest, ApprovalDecision)


def check_sealable(flow: str, encoder: Encoder | None, **parts: Any) -> None:
    """Refuse values a pause could not seal, before an approval record is created.

    Raises:
        FlowError: `state-not-sealable`, naming the path and type.
    """
    try:
        for name, value in parts.items():
            encode(value, encoder, name)
    except NotSealable as exc:
        raise not_sealable(flow, exc) from exc


def not_sealable(flow: str, exc: NotSealable) -> FlowError:
    """The FlowError for a value that cannot be sealed."""
    logger.warning("flow %s: %s at %s (%s)", flow, NOT_SEALABLE, exc.path, exc.type_name)
    return FlowError(
        f"flow {flow!r}: {NOT_SEALABLE}: {exc} (pass FlowRunner(seal_encoder=...), or keep the "
        "value out of the state)",
        flow=flow,
    )


def payload(checkpoint: FlowCheckpoint, encoder: Encoder | None) -> dict[str, Any]:
    """Every field but the MAC, in the canonical encoding.

    Raises:
        NotSealable: a value has no canonical encoding.
    """
    journal = {node: data for node, data in checkpoint._journal.items()}
    check_plain(journal, "journal")
    data = {
        "flow": checkpoint.flow,
        "graph": checkpoint._graph,
        "state": encode(checkpoint.state, encoder, "state"),
        "frontier": encode(tuple(checkpoint.frontier)),
        "visits": encode(dict(checkpoint.visits)),
        "loops": encode(dict(checkpoint.loops)),
        "steps": encode(checkpoint.steps),
        "pending": encode(checkpoint.pending, encoder, "pending"),
        "decisions": encode(dict(checkpoint.decisions), encoder, "decisions"),
        "writes": encode(
            {n: tuple(dict(u) for u in v) for n, v in checkpoint._writes.items()}, encoder, "writes"
        ),
        "owned": encode(checkpoint._owned),
        "answers": encode(
            {n: tuple(tuple(e) for e in v) for n, v in checkpoint._answers.items()}, encoder
        ),
        "items": encode({n: dict(v) for n, v in checkpoint._items.items()}, encoder, "items"),
        "journal": journal,
    }
    if checkpoint._children:  # absent when empty, so checkpoints without sub-flows seal as before
        data["children"] = {n: payload(c, encoder) for n, c in checkpoint._children.items()}
    return data


def mac(key: bytes, data: Mapping[str, Any]) -> str:
    """HMAC-SHA256 hex of the canonical JSON of a payload."""
    return hmac.new(key, canonical_bytes(data), hashlib.sha256).hexdigest()


def seal(sealer: Sealer, checkpoint: FlowCheckpoint) -> str:
    """The MAC of a checkpoint.

    Raises:
        NotSealable: a value has no canonical encoding.
    """
    return mac(sealer.key, payload(checkpoint, sealer.encoder))


def check_seal(sealer: Sealer, checkpoint: FlowCheckpoint) -> None:
    """Refuse a checkpoint this runner did not seal, or one changed since.

    Raises:
        FlowError: the MAC is missing, not a str, or does not match.
    """
    try:
        ok = (
            isinstance(checkpoint._mac, str)
            and bool(checkpoint._mac)
            and hmac.compare_digest(checkpoint._mac, seal(sealer, checkpoint))
        )
    except (NotSealable, TypeError, ValueError):
        ok = False
    if not ok:
        raise tampered(
            checkpoint.flow,
            "the checkpoint was not sealed with this runner's key or was changed since",
        )


def tampered(flow: str, why: str) -> FlowError:
    """The FlowError for a checkpoint that must not resume."""
    logger.warning("flow %s: %s (%s)", flow, TAMPERED, why)
    return FlowError(f"flow {flow!r}: {TAMPERED}: {why}", flow=flow)


def check_graph(graph: FlowGraph, checkpoint: FlowCheckpoint) -> None:
    """Refuse a checkpoint paused in another graph with the same flow name.

    Raises:
        FlowError: the graph digest differs.
    """
    if checkpoint._graph != graph_digest(graph):
        logger.warning("flow %s: checkpoint is for another graph", graph.name)
        raise FlowError(
            f"flow {graph.name!r}: the checkpoint was paused in another version of this flow "
            "(its nodes, edges or gates differ); resume refused",
            flow=graph.name,
        )


def dump_json(sealer: Sealer, checkpoint: FlowCheckpoint) -> str:
    """The checkpoint as JSON text: format, payload and MAC.

    Raises:
        FlowError: the checkpoint is not sealed by this runner (`checkpoint-tampered`).
    """
    check_seal(sealer, checkpoint)
    data = payload(checkpoint, sealer.encoder)
    if sealer.decoder is None:
        _check_loadable(checkpoint, data)
    document = {"format": FORMAT, "payload": data, "mac": checkpoint._mac}
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _check_loadable(checkpoint: FlowCheckpoint, data: Mapping[str, Any]) -> None:
    """Refuse to dump a class that `load_json` would refuse to build."""
    for node, child in checkpoint._children.items():
        _check_loadable(child, data["children"][node])
    allowed = allowed_classes(type(checkpoint.state))
    # replays check their own classes; sub-flow runs were checked above with their own model
    parts = {k: v for k, v in data.items() if k not in ("journal", "children")}
    missing = sorted(tagged_classes(list(parts.values())) - allowed.keys())
    if missing:
        logger.warning(
            "flow %s: checkpoint holds class %s; not dumped", checkpoint.flow, missing[0]
        )
        raise FlowError(
            f"flow {checkpoint.flow!r}: {NOT_SEALABLE}: the checkpoint holds {missing[0]}, which "
            "the state model's annotations do not reach, so it could not be loaded; annotate the "
            "field with that type or pass FlowRunner(seal_decoder=...)",
            flow=checkpoint.flow,
        )


def load_json(sealer: Sealer, graph: FlowGraph, text: str | bytes) -> FlowCheckpoint:
    """A checkpoint from `dump_json` text; the MAC is checked before anything is built.

    Raises:
        FlowError: not a checkpoint document, a wrong MAC (`checkpoint-tampered`), another
            graph, or a value that cannot be loaded.
    """
    try:
        document = json.loads(text)
    except (ValueError, RecursionError) as exc:  # RecursionError: nested too deeply
        raise tampered(graph.name, "not JSON") from exc
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise tampered(graph.name, "not a flow checkpoint document")
    data, given = document.get("payload"), document.get("mac")
    if not isinstance(data, dict) or not isinstance(given, str):
        raise tampered(graph.name, "payload or MAC missing")
    if not hmac.compare_digest(given, mac(sealer.key, data)):
        raise tampered(graph.name, "the MAC does not match")
    if data.get("graph") != graph_digest(graph):
        raise FlowError(
            f"flow {graph.name!r}: the checkpoint was paused in another version of this flow",
            flow=graph.name,
        )
    try:
        checkpoint = _build(sealer, graph, data, given)
    except (NotSealable, KeyError, TypeError, ValueError) as exc:
        log_failure(
            logger, logging.WARNING, exc, "flow %s: checkpoint cannot be loaded", graph.name
        )
        raise FlowError(
            f"flow {graph.name!r}: checkpoint cannot be loaded: {exc}", flow=graph.name
        ) from exc
    check_seal(sealer, checkpoint)  # the rebuilt objects seal back to the same MAC
    return checkpoint


def _build(sealer: Sealer, graph: FlowGraph, data: dict[str, Any], given: str) -> FlowCheckpoint:
    model = graph.state_model
    allowed = allowed_classes(model)

    def load(part: str) -> Any:
        return decode(data[part], sealer.decoder, allowed)

    state = load("state")
    if not isinstance(state, model):
        raise TypeError(f"state is {type(state).__name__}, not {model.__name__}")
    pending = load("pending")
    if pending is not None and not isinstance(pending, ApprovalRequest):
        raise TypeError("pending is not an ApprovalRequest")
    decisions = load("decisions")
    if not all(isinstance(d, ApprovalDecision) for d in decisions.values()):
        raise TypeError("decisions are not ApprovalDecisions")
    children = {}
    for node, part in dict(data.get("children", {})).items():
        spec = graph.nodes.get(node)
        if spec is None or spec.child is None or not isinstance(part, dict):
            raise TypeError(f"{node!r} is not a SubFlow node of this graph")
        children[node] = _build(sealer, spec.child, part, "")
    if data["flow"] != graph.name:
        raise TypeError(f"the checkpoint is for flow {data['flow']!r}")
    return FlowCheckpoint(
        flow=data["flow"],
        state=state,
        frontier=load("frontier"),
        visits=frozen(load("visits")),
        loops=frozen(load("loops")),
        steps=load("steps"),
        pending=pending,
        decisions=frozen(decisions),
        _writes=frozen(load("writes")),
        _owned=load("owned"),
        _answers=frozen(load("answers")),
        _items=frozen({n: frozen(v) for n, v in load("items").items()}),
        _journal=frozen(data["journal"]),
        _children=frozen(children),
        _graph=data["graph"],
        _mac=given,
    )
