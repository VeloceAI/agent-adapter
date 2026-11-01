"""Formats that framework integrations add: `register_format`, read by `convert` and "auto"."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from agent_adapter.context.formats._shared import Converter
from agent_adapter.core.errors import ConfigError

log = logging.getLogger(__name__)
Detect = Callable[[Any], bool]  # one native item -> it is a message of this format
BUILT_IN = frozenset({"auto", "agent_adapter", "langchain", "openai", "anthropic", "gemini"})


@dataclass(frozen=True)
class FormatSpec:
    """A registered format: its converter and, for "auto", a per-item test."""

    name: str
    converter: type[Converter]
    detect: Detect | None


_FORMATS: dict[str, FormatSpec] = {}
_LOCK = threading.Lock()


def register_format(name: str, converter: type[Converter], *, detect: Detect | None = None) -> None:
    """Make `format=name` use `converter`; with `detect`, "auto" finds it too.

    Integrations call this when they are imported, so `context` never imports a framework.
    Registering the same converter class again (a module reload) replaces it.

    Raises:
        ConfigError: the name is empty or built in, the converter is not a `Converter`,
            `detect` is not callable, or another converter already has the name.
    """
    if not name or name in BUILT_IN:
        raise ConfigError(f"cannot register the message format {name!r}: empty or built in")
    if not (isinstance(converter, type) and issubclass(converter, Converter)):
        raise ConfigError(f"the {name!r} converter must be a Converter subclass")
    if detect is not None and not callable(detect):
        raise ConfigError(f"detect for the {name!r} format must be callable(item) -> bool")
    with _LOCK:
        old = _FORMATS.get(name)
        if old is not None and _qualname(old.converter) != _qualname(converter):
            raise ConfigError(
                f"the message format {name!r} is already registered by {_qualname(old.converter)}"
            )
        _FORMATS[name] = FormatSpec(name, converter, detect)
    log.debug("message format registered", extra={"format": name})


def registered(name: str) -> FormatSpec | None:
    """The registered format called `name`, if any."""
    return _FORMATS.get(name)


def registered_names() -> tuple[str, ...]:
    """Names of the registered formats, in registration order."""
    return tuple(_FORMATS)


def detect_registered(items: Sequence[Any]) -> str | None:
    """The first registered format whose `detect` accepts every item, if any."""
    for spec in list(_FORMATS.values()):
        if spec.detect is not None and items and all(spec.detect(item) for item in items):
            return spec.name
    return None


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"
