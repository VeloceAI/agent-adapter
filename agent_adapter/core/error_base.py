"""The root of the error hierarchy, and the helpers that name and log failures."""

from __future__ import annotations

import logging
from typing import Any, ClassVar


class AdapterError(Exception):
    """Base class for every error raised by Agent Adapter."""

    __module__ = "agent_adapter.core.errors"  # the public home: stable names in logs and pickles

    kind: ClassVar[str] = "internal-error"  # stable id in ErrorInfo, HTTP problems, events
    retryable: ClassVar[bool] = False  # another attempt may succeed

    def __reduce__(self) -> tuple[Any, ...]:
        # Keyword-only constructors break the default `cls(*args)` unpickling.
        return (_restore, (type(self), self.args, self.__dict__))


def _restore(cls: type[AdapterError], args: tuple[Any, ...], state: dict[str, Any]) -> AdapterError:
    error = cls.__new__(cls)
    error.args = args
    error.__dict__.update(state)
    return error


def qualified_name(cls: type) -> str:
    """`module.QualName` of a class; builtins without the module."""
    if cls.__module__ == "builtins":
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


def log_failure(
    log: logging.Logger, level: int, exc: BaseException, message: str, *args: object
) -> None:
    """Log `message` plus the exception's type at `level`; its text and traceback at DEBUG.

    Exception text can carry tool arguments, results or credentials, so it stays at DEBUG.
    """
    log.log(level, message + " (%s)", *args, qualified_name(type(exc)), stacklevel=2)
    log.debug(message, *args, exc_info=exc, stacklevel=2)


_restore.__module__ = "agent_adapter.core.errors"  # pickles name the public home
