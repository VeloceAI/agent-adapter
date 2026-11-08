"""Per-node cache: a key from the node and the inputs it reads, entries as signed canonical JSON."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import hmac
import inspect
import json
import logging
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import FlowError, log_failure
from agent_adapter.core.ports import CachePort, EventSink
from agent_adapter.core.types import Event
from agent_adapter.exec._canon import (
    Decoder,
    Encoder,
    NotSealable,
    canonical_bytes,
    check_plain,
    decode,
    digest,
    encode,
)
from agent_adapter.flow._digest import action_shape, identity
from agent_adapter.flow._invoke import describe_error
from agent_adapter.flow._updates import Updates
from agent_adapter.flow.actions import Call, Join, Map, NodeCache, SubFlow
from agent_adapter.flow.graph import NodeSpec
from agent_adapter.tracing.events import EventType, emit, make_event
from agent_adapter.tracing.sinks import FanoutSink, first_sink

logger = logging.getLogger(__name__)

NAMESPACE = "aa"
FORMAT = 1
# Without a runner key, entries are only trusted by the process that wrote them.
_PROCESS_KEY = secrets.token_bytes(32)
_ASKED = frozenset(
    {
        EventType.APPROVAL_REQUESTED,
        EventType.APPROVAL_DECIDED,
        EventType.TOOL_PENDING,
        EventType.NODE_PAUSED,
    }
)
_WARNED: set[tuple[str, ...]] = set()  # (flow, node[, why]) told once: not cached


class ApprovalWatch:
    """A sink leaf that notes whether a node visit asked for any approval."""

    def __init__(self) -> None:
        self.asked = False

    def emit(self, event: Event) -> None:
        """Note approval and pause events."""
        if event.type in _ASKED:
            self.asked = True


def watched(ctx: ToolContext, fallback: EventSink | None) -> tuple[ToolContext, ApprovalWatch]:
    """`ctx` with its events also going to a new `ApprovalWatch`."""
    watch = ApprovalWatch()
    inner = first_sink(ctx.sink, fallback)
    sink = FanoutSink(inner, watch) if inner is not None else FanoutSink(watch)
    return dataclasses.replace(ctx, sink=sink), watch


@dataclass(frozen=True)
class NodeCacheStore:
    """One node visit's view of the cache port: its key, signing key and codec."""

    port: CachePort
    key: str
    secret: bytes
    encoder: Encoder | None
    decoder: Decoder | None
    allowed: Mapping[str, type]

    async def get(self, flow: str, node: str) -> Updates | None:
        """The cached updates, or None on a miss, a bad or unsigned entry, or a backend error."""
        try:
            data = await self.port.get(self.key)
        except Exception as exc:
            if is_control_flow(exc):
                raise
            log_failure(logger, logging.WARNING, exc, "node cache get failed for %s.%s", flow, node)
            return None
        if data is None:
            logger.debug("node cache miss for %s.%s", flow, node)
            return None
        try:
            return self._read(data)
        except (NotSealable, ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
            log_failure(
                logger, logging.WARNING, exc, "node cache entry for %s.%s ignored", flow, node
            )
            return None

    def _read(self, data: bytes) -> Updates:
        document = json.loads(data.decode("utf-8"))
        if not isinstance(document, dict) or document.get("v") != FORMAT:
            raise ValueError("not a node cache entry")
        body, given = document.get("body"), document.get("mac")
        if not isinstance(given, str) or not hmac.compare_digest(given, self._mac(body)):
            raise ValueError("the entry's MAC does not match")
        check_plain(body)
        if not isinstance(body, dict):
            raise TypeError("the entry has no body")
        items = decode(body["items"], self.decoder, self.allowed)
        view = decode(body["view"], self.decoder, self.allowed)
        if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
            raise TypeError("cached updates are not mappings")
        if view is not None and not isinstance(view, dict):
            raise TypeError("cached view is not a mapping")
        return Updates(items, view, cached=True)

    async def put(
        self,
        flow: str,
        node: str,
        items: Sequence[Mapping[str, Any]],
        view: Mapping[str, Any] | None,
        ttl: float | None,
    ) -> bool:
        """Store a finished visit; False when a value has no canonical form or the port fails."""
        try:
            body = {
                "items": encode([dict(i) for i in items], self.encoder, "items"),
                "view": encode(None if view is None else dict(view), self.encoder, "view"),
            }
            data = canonical_bytes({"v": FORMAT, "body": body, "mac": self._mac(body)})
        except NotSealable as exc:
            logger.debug("node %s.%s not cached: %s at %s", flow, node, exc.type_name, exc.path)
            return False
        try:
            await self.port.set(self.key, data, ttl)
        except Exception as exc:
            if is_control_flow(exc):
                raise
            log_failure(logger, logging.WARNING, exc, "node cache set failed for %s.%s", flow, node)
            return False
        return True

    def _mac(self, body: Any) -> str:
        # The key (scope, owner, flow, node, input digest) is signed too: an entry copied to
        # another key, scope or tenant does not verify.
        signed = canonical_bytes({"key": self.key, "body": body})
        return hmac.new(self.secret, signed, hashlib.sha256).hexdigest()


def cache_store(
    node: NodeSpec,
    flow: str,
    state: BaseModel,
    ctx: ToolContext,
    *,
    secret: bytes,
    encoder: Encoder | None,
    decoder: Decoder | None,
    allowed: Mapping[str, type],
) -> NodeCacheStore | None:
    """The visit's cache entry handle; None when there is no port or no key for these inputs.

    Raises:
        FlowError: the node's `key` function failed.
    """
    cache: NodeCache = node.cache
    if ctx.cache is None:
        if (flow, node.name) not in _WARNED:
            _WARNED.add((flow, node.name))
            logger.warning("flow %s node %s: cache= but no cache port; not cached", flow, node.name)
        return None
    owner = {"run": ctx.run_id, "thread": ctx.thread_id}.get(cache.scope, "global")
    if not owner:
        logger.debug("node %s.%s: no %s id, so not cached", flow, node.name, cache.scope)
        return None
    try:
        config = digest(bound_config(node), encoder)
    except NotSealable as exc:
        if cache.key is None:
            if (flow, node.name, "config") not in _WARNED:
                _WARNED.add((flow, node.name, "config"))
                logger.warning(
                    "flow %s node %s: its code captures values with no canonical form (%s); "
                    "not cached (pass cache=NodeCache(key=...) covering them)",
                    flow,
                    node.name,
                    exc.type_name,
                )
            return None
        config = "key"  # the user's key= covers what the code captured
    material = [flow, node.name, action_shape(node), config, _inputs(cache, flow, node.name, state)]
    try:
        hashed = digest(material, encoder)
    except NotSealable as exc:
        logger.debug("node %s.%s: inputs have no canonical form (%s)", flow, node.name, exc.path)
        return None
    where = "global" if cache.scope == "global" else f"{cache.scope}={owner}"
    key = f"{NAMESPACE}:node:v{FORMAT}:{where}:{flow}:{node.name}:{hashed}"
    return NodeCacheStore(ctx.cache, key, secret or _PROCESS_KEY, encoder, decoder, allowed)


def _inputs(cache: NodeCache, flow: str, node: str, state: BaseModel) -> Any:
    if cache.reads is not None:
        return {name: getattr(state, name) for name in cache.reads}
    if cache.key is None:
        return state
    try:
        return {"key": cache.key(state)}
    except Exception as exc:
        if is_control_flow(exc):
            raise
        raise FlowError(
            f"flow {flow!r} node {node!r}: cache key failed: {describe_error(exc)}",
            flow=flow,
            node=node,
        ) from exc


@dataclass(frozen=True)
class Codec:
    """How entries are signed and encoded: the runner's key and seal codec, the state classes."""

    secret: bytes
    encoder: Encoder | None
    decoder: Decoder | None
    allowed: Mapping[str, type]


@dataclass(frozen=True)
class CachedVisit:
    """One visit of a cached node: look up its entry, else run it and store what it made."""

    node: NodeSpec
    flow: str
    ctx: ToolContext
    ids: Mapping[str, Any]
    watch: ApprovalWatch | None
    codec: Codec

    async def run(self, state: BaseModel, attempt: Callable[[], Awaitable[Updates]]) -> Updates:
        """The cached updates (after a `node.cached` event), else `attempt()`'s, stored.

        Raises:
            FlowError: the node's `key` function failed; as `attempt`.
        """
        codec, name = self.codec, self.node.name
        store = cache_store(
            self.node,
            self.flow,
            state,
            self.ctx,
            secret=codec.secret,
            encoder=codec.encoder,
            decoder=codec.decoder,
            allowed=codec.allowed,
        )
        if store is None:
            return await attempt()
        hit = await store.get(self.flow, name)
        if hit is not None:
            emit(self.ctx.sink, make_event(EventType.NODE_CACHED, self.ctx, **self.ids))
            logger.debug("flow %s node %s answered from the cache", self.flow, name)
            return hit
        done = await attempt()
        if self.watch is not None and self.watch.asked:
            logger.warning("flow %s node %s asked for approval; not cached", self.flow, name)
        else:
            await store.put(self.flow, name, done.items, done.view, self.node.cache.ttl)
        return done


def bound_config(node: NodeSpec) -> list[Any]:
    """What a node's code captured outside the state: closure cells, partial and bound arguments.

    Two flows whose functions share a name but captured other values get other keys. Static
    `Call` arguments count too; a sub-flow's child nodes are included.
    """
    found: list[Any] = []
    for fn in _callables(node.action):
        found.append(_captured(fn, 0))
    if isinstance(node.action, Call) and not callable(node.action.args):
        found.append({"args": dict(node.action.args)})
    if node.child is not None:
        found.extend(bound_config(child) for child in node.child.nodes.values())
    return found


def _callables(action: Any) -> list[Any]:
    if isinstance(action, Call):
        return [action.args] if callable(action.args) else []
    if isinstance(action, Map):
        return [fn for fn in (action.args_fn, action.fn) if fn is not None]
    if isinstance(action, Join):
        return [action.fn] if action.fn is not None else []
    if isinstance(action, SubFlow):
        return [spec for spec in (action.input, action.into) if callable(spec)]
    return [action] if callable(action) else []


_MAX_DEPTH = 4  # closures of closures: deeper ones count by identity only


def _captured(fn: Any, depth: int) -> Any:
    """`fn`'s identity and every value bound into it: closure cells, defaults, partial
    arguments, a bound `self` or a callable object's attributes (NotSealable on encoding)."""
    if depth > _MAX_DEPTH:
        return identity(fn)
    if isinstance(fn, functools.partial):
        return {
            "partial": _captured(fn.func, depth + 1),
            "args": [_value(v, depth) for v in fn.args],
            "kw": {k: _value(v, depth) for k, v in fn.keywords.items()},
        }
    if inspect.ismethod(fn):
        return {"method": _captured(fn.__func__, depth + 1), "self": _state(fn.__self__)}
    if inspect.isfunction(fn):
        cells = []
        for cell in fn.__closure__ or ():
            try:
                value = cell.cell_contents
            except ValueError:  # an empty cell: nothing captured yet
                value = None
            cells.append(_value(value, depth))
        return {
            "fn": identity(fn),
            "cells": cells,
            "defaults": [_value(v, depth) for v in fn.__defaults__ or ()],
            "kwdefaults": {k: _value(v, depth) for k, v in (fn.__kwdefaults__ or {}).items()},
        }
    if isinstance(fn, type) or inspect.isbuiltin(fn):
        return identity(fn)
    call = type(fn).__call__  # a callable object: its attributes and its __call__'s defaults
    return {"object": identity(fn), "state": _state(fn), "call": _captured(call, depth + 1)}


def _value(value: Any, depth: int) -> Any:
    """A captured value: code by what it captured, a class by identity, data as itself."""
    if isinstance(value, type):
        return identity(value)
    if callable(value):
        return _captured(value, depth + 1)
    return value


def _state(obj: Any) -> Any:
    """An object's own state: a model as itself, else its attributes (`__dict__`)."""
    if isinstance(obj, BaseModel) or not hasattr(obj, "__dict__"):
        return obj
    return {"class": identity(type(obj)), "attrs": dict(vars(obj))}
