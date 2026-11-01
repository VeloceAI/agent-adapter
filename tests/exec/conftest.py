"""Fixtures for the executor tests."""

from collections.abc import Iterator

import pytest

from agent_adapter.core._default import reset_default_registry
from agent_adapter.exec.defaults import reset_default_executor


@pytest.fixture
def isolated_defaults() -> Iterator[None]:
    """An empty default registry and a fresh default executor; tools restored afterwards."""
    reset_default_executor()
    restore = reset_default_registry()
    yield
    restore()
    reset_default_executor()
