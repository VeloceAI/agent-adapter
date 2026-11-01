"""Load `module.path:attribute` targets named on the command line; never a file path."""

from __future__ import annotations

import importlib
import logging
import re
from typing import Any

from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

TARGET = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


def load_target(text: str, option: str) -> Any:
    """The attribute a `module.path:attribute` string names.

    Raises:
        ConfigError: a malformed target, a module that fails to import, or no such attribute;
            the message names `option` and the target.
    """
    target = str(text or "")
    if not TARGET.fullmatch(target):
        raise ConfigError(
            f"{option} {target!r}: use module.path:attribute (for example app.chat:agent)"
        )
    module_name, attribute = target.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.debug("chat: importing %s for %s failed", module_name, option, exc_info=True)
        raise ConfigError(
            f"{option} {target!r}: cannot import {module_name} ({type(exc).__name__}: {exc})"
        ) from exc
    try:
        return getattr(module, attribute)
    except AttributeError:
        raise ConfigError(f"{option} {target!r}: {module_name} has no {attribute!r}") from None
