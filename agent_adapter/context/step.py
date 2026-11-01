"""The compression step interface: the protocol, a base class and the context steps receive."""

from __future__ import annotations

import dataclasses
import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from agent_adapter.cache.keys import digest
from agent_adapter.context.conversation import Protection, Unit, split_units
from agent_adapter.context.tokens import TokenMeter
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from agent_adapter.core.ports import CachePort


@dataclass
class StepContext:
    """What a step may read during one `compress` call; the same object for every step."""

    target_tokens: int  # stop as soon as the conversation is at or under this
    meter: TokenMeter
    protection: Protection
    ctx: ToolContext | None = None  # the run the compression belongs to, when known
    summary_cost_usd: float = 0.0  # AI steps add what their model calls cost
    cache: CachePort | None = None  # the policy's cache, for steps without their own

    def tokens(self, messages: Sequence[Message]) -> int:
        """Tokens of the whole list, with the policy's counter."""
        return self.meter.total(messages)

    def message_tokens(self, message: Message) -> int:
        """Tokens of one message (cached); steps keep a running total with these."""
        return self.meter.of(message)

    def is_protected(self, message: Message) -> bool:
        """True when the step must not change or remove `message`."""
        return self.protection.is_protected(message)

    def units(self, messages: Sequence[Message]) -> list[Unit]:
        """`messages` as atomic units, each flagged protected or not."""
        return split_units(messages, self.protection)

    def under_target(self, tokens: int) -> bool:
        """True once `tokens` is at or under the target: the step must stop."""
        return tokens <= self.target_tokens

    def add_summary_cost(self, usd: float) -> None:
        """Record what a model call made by a step cost.

        Raises:
            ConfigError: `usd` is negative or not a finite number.
        """
        if isinstance(usd, bool) or not isinstance(usd, int | float) or not math.isfinite(usd):
            raise ConfigError("a summary cost must be a finite number")
        if usd < 0:
            raise ConfigError("a summary cost must be >= 0")
        self.summary_cost_usd += float(usd)


@runtime_checkable
class CompressionStep(Protocol):
    """One stage of a compression pipeline.

    `apply` returns the new message list. It must stop as soon as the conversation is under
    `context.target_tokens`, and should leave protected messages and tool rounds alone; the
    invariant guard reverts any change that breaks a rule anyway.
    """

    @property
    def name(self) -> str: ...  # snake_case; used in reports, logs and cache keys

    async def apply(self, messages: list[Message], context: StepContext) -> list[Message]: ...


class BaseCompressionStep(ABC):
    """Base class for steps: a fixed `name`, `apply`, and `config()` for cache keys and logs."""

    name: ClassVar[str] = "step"
    # True only on steps with an `include_system_prompt` setting (cut, summarise, your own).
    may_include_system_prompt: ClassVar[bool] = False

    @abstractmethod
    async def apply(self, messages: list[Message], context: StepContext) -> list[Message]:
        """The compressed list; see `CompressionStep`."""

    def config(self) -> dict[str, Any]:
        """The step's settings as JSON-safe values; objects and functions by their identity."""
        settings: dict[str, Any] = {"step": self.name}
        if dataclasses.is_dataclass(self):
            for field in dataclasses.fields(self):
                settings[field.name] = _plain(getattr(self, field.name))
        return settings


def shortens_system_prompt(step: object) -> bool:
    """True when `step` may shorten the system message: its class allows it and it opted in."""
    allowed = getattr(type(step), "may_include_system_prompt", False) is True
    return allowed and getattr(step, "include_system_prompt", False) is True


def check_flag(name: str, value: object) -> None:
    """Validate a step's bool setting.

    Raises:
        ConfigError: `value` is not a bool.
    """
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a bool, got {value!r}")


def check_ttl(name: str, value: object) -> None:
    """Validate a lifetime in seconds.

    Raises:
        ConfigError: `value` is not None or a finite number > 0.
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be a number of seconds > 0 or None, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{name} must be a number of seconds > 0 or None, got {value!r}")


def check_count(name: str, value: object, *, minimum: int) -> None:
    """Validate a step's int setting.

    Raises:
        ConfigError: `value` is not an int (bools excluded) or is below `minimum`.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an int >= {minimum}, got {value!r}")


_SCALARS = (str, int, float, bool, bytes, type(None))
_NAME_ATTRS = ("provider", "model", "api")  # what tells one model, embedder or reranker apart
_INNER_ATTRS = ("embedder", "reranker", "scorer", "wrapped")
_FN_ATTRS = ("fn", "_fn")
_MAX_DEPTH = 3


def _plain(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_plain(item, depth) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item, depth) for key, item in value.items()}
    if isinstance(value, type):
        return _qualname(value)
    if inspect.isroutine(value):
        return routine_identity(value)
    return _object_identity(value, depth)


def routine_identity(fn: Any) -> str:
    """`module.qualname` of a function plus a digest of its code, so two lambdas differ."""
    target = getattr(fn, "__func__", fn)  # a bound method's function
    name = f"{getattr(target, '__module__', '')}.{getattr(target, '__qualname__', '')}"
    code = getattr(target, "__code__", None)
    if code is None:
        return name
    cells = [cell.cell_contents for cell in getattr(target, "__closure__", None) or ()]
    material = [
        code.co_code.hex(),
        [_scalar(const) for const in code.co_consts],
        list(code.co_names),
        [_scalar(cell) for cell in cells],
    ]
    return f"{name}#{digest(material)[:12]}"


def _scalar(value: Any) -> Any:
    """A JSON-safe, process-independent stand-in for a code constant or closure value."""
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, tuple | frozenset):
        return [_scalar(item) for item in value]
    if inspect.iscode(value):
        return value.co_code.hex()
    return _qualname(type(value))


def _object_identity(value: Any, depth: int) -> dict[str, Any]:
    """An object as its type plus what tells instances apart (names, key fields, functions)."""
    out: dict[str, Any] = {"type": _qualname(type(value))}
    for attr in _NAME_ATTRS:
        found = getattr(value, attr, None)
        if isinstance(found, str):
            out[attr] = found
    fields = getattr(value, "cache_key_fields", None)
    if callable(fields):
        out["fields"] = _plain(fields(), depth + 1)
    for attr in _FN_ATTRS:
        found = getattr(value, attr, None)
        if callable(found):
            out["fn"] = (
                routine_identity(found) if inspect.isroutine(found) else _qualname(type(found))
            )
    if depth < _MAX_DEPTH:
        for attr in _INNER_ATTRS:
            found = getattr(value, attr, None)
            if found is not None:
                out[attr] = _plain(found, depth + 1)
    return out


def plain(value: Any) -> Any:
    """`value` as JSON-safe cache-key material; objects and functions by their identity."""
    return _plain(value)


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"
