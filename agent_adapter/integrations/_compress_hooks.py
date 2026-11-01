"""Shared pieces of the framework compression hooks: one `Shrink` and a model wrapper."""

from __future__ import annotations

import copy
import functools
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from agent_adapter.context.formats import Pinned, compress_for, compress_for_sync
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)
Kind = Literal["call", "acall", "iter", "aiter"]  # sync, async, generator, async generator
_WRAPPED = "_agent_adapter_shrink"


@dataclass(frozen=True)
class Shrink:
    """Compress one framework's native message list with a fixed policy."""

    policy: CompressionPolicy
    format: str
    pinned: Pinned | None = None
    model_max_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy, CompressionPolicy):
            raise ConfigError(f"policy must be a CompressionPolicy, got {type(self.policy)!r}")
        if self.pinned is not None and not callable(self.pinned):
            raise ConfigError("pinned must be a callable(native message) -> bool")

    async def __call__(self, items: Sequence[Any]) -> list[Any]:
        """The compressed list, in the same native type.

        Raises:
            ConfigError: as `compress_for`.
        """
        smaller, report = await compress_for(
            items,
            self.policy,
            format=self.format,
            pinned=self.pinned,
            model_max_tokens=self.model_max_tokens,
        )
        self._log(report.messages_before, report.messages_after, report.tokens_after)
        return smaller

    def sync(self, items: Sequence[Any]) -> list[Any]:
        """`__call__` for sync framework code.

        Raises:
            ConfigError: as `compress_for_sync`.
        """
        smaller, report = compress_for_sync(
            items,
            self.policy,
            format=self.format,
            pinned=self.pinned,
            model_max_tokens=self.model_max_tokens,
        )
        self._log(report.messages_before, report.messages_after, report.tokens_after)
        return smaller

    def _log(self, before: int, after: int, tokens: int) -> None:
        logger.debug(
            "framework model input compressed",
            extra={"format": self.format, "messages": [before, after], "tokens_after": tokens},
        )


def wrap_model(model: Any, shrink: Shrink, methods: Mapping[str, Kind]) -> Any:
    """A shallow copy of `model` whose `methods` get compressed `messages`.

    The copy's class is a subclass with the same name, so a framework that saves the model
    by class name loads the plain one back. Methods the class lacks stay missing.

    Raises:
        ConfigError: the class has none of `methods`.
    """
    base = type(model)
    if getattr(base, _WRAPPED, None) is not None:
        base = base.__mro__[1]  # wrapping again replaces the policy, never stacks it
    namespace: dict[str, Any] = {_WRAPPED: shrink}
    for name, kind in methods.items():
        original = getattr(base, name, None)
        if callable(original):
            namespace[name] = _WRAPPERS[kind](original, shrink)
    if len(namespace) == 1:
        raise ConfigError(f"{base.__name__} has none of {sorted(methods)}; it is not a model")
    wrapped = copy.copy(model)
    wrapped.__class__ = type(base.__name__, (base,), namespace)
    return wrapped


def _swap(args: tuple[Any, ...], kwargs: dict[str, Any], new: Callable[[Any], Any]) -> Any:
    """(args, kwargs) with `messages` (keyword or first positional) replaced by `new(messages)`."""
    if "messages" in kwargs:
        return args, {**kwargs, "messages": new(kwargs["messages"])}
    if args:
        return (new(args[0]), *args[1:]), kwargs
    return args, kwargs


async def _aswap(args: tuple[Any, ...], kwargs: dict[str, Any], shrink: Shrink) -> Any:
    if "messages" in kwargs:
        return args, {**kwargs, "messages": await shrink(kwargs["messages"])}
    if args:
        return (await shrink(args[0]), *args[1:]), kwargs
    return args, kwargs


def _call(original: Callable[..., Any], shrink: Shrink) -> Callable[..., Any]:
    @functools.wraps(original)
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        args, kwargs = _swap(args, kwargs, shrink.sync)
        return original(self, *args, **kwargs)

    return method


def _acall(original: Callable[..., Any], shrink: Shrink) -> Callable[..., Any]:
    @functools.wraps(original)
    async def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        args, kwargs = await _aswap(args, kwargs, shrink)
        return await original(self, *args, **kwargs)

    return method


def _iter(original: Callable[..., Any], shrink: Shrink) -> Callable[..., Any]:
    @functools.wraps(original)
    def method(self: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        args, kwargs = _swap(args, kwargs, shrink.sync)
        yield from original(self, *args, **kwargs)

    return method


def _aiter(original: Callable[..., Any], shrink: Shrink) -> Callable[..., Any]:
    @functools.wraps(original)
    async def method(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        args, kwargs = await _aswap(args, kwargs, shrink)
        async for event in original(self, *args, **kwargs):
            yield event

    return method


_WRAPPERS: dict[Kind, Callable[[Callable[..., Any], Shrink], Callable[..., Any]]] = {
    "call": _call,
    "acall": _acall,
    "iter": _iter,
    "aiter": _aiter,
}
