"""Versioning guide: check that a name you use is public, and find where moved names live."""

import importlib

import agent_adapter


# --8<-- [start:public]
def is_public(module: str, name: str) -> bool:
    """True when `name` is in the `__all__` of a public agent_adapter module."""
    if any(part.startswith("_") for part in module.split(".")) or ".core" in module:
        return False  # private modules and agent_adapter.core are internal
    return name in getattr(importlib.import_module(module), "__all__", ())


def moved(name: str) -> str:
    """The message a pre-1.0 top-level name gives today."""
    try:
        getattr(agent_adapter, name)
    except AttributeError as exc:
        return str(exc)
    return "still top level"


# --8<-- [end:public]


def main() -> None:
    # --8<-- [start:check]
    print(len(agent_adapter.__all__), "top-level names")  # plus every subpackage's __all__
    print(is_public("agent_adapter", "Executor"), is_public("agent_adapter.exec", "Backoff"))
    print(is_public("agent_adapter.core.errors", "log_failure"))  # internal: may change
    print(moved("ToolNotFound"))
    print(moved("ShortCircuit"))
    # --8<-- [end:check]


if __name__ == "__main__":
    main()
