"""The process-wide default registry behind `agent_adapter.tool` and `agent_adapter.wrap`."""

from collections.abc import Callable

from agent_adapter.core.registry import Registry

# Module import runs once under the import lock, so every thread sees the same registry.
default_registry = Registry()
tool = default_registry.tool
wrap = default_registry.wrap


def reset_default_registry() -> Callable[[], None]:
    """Unregister every tool of the default registry; return a function that restores them.

    `Executor()` with no registry reaches every `@tool` of every imported module. Tests and
    servers that want a known set call this, then register only what they serve.
    """
    saved = list(default_registry)
    for spec in saved:
        default_registry.remove(spec.name)

    def restore() -> None:
        for spec in list(default_registry):
            default_registry.remove(spec.name)
        default_registry.add_many(saved)

    return restore
