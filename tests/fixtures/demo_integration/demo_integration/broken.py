"""A plugin module that fails on import; discovery must skip it with a warning."""

raise RuntimeError("demo_integration.broken is broken on purpose")
