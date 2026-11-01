"""Suite-wide setup."""

import sys
from collections.abc import Iterator

import pytest

from agent_adapter.models._port import warm_platform_cache

# Fill platform's caches without WMI before any test or SDK queries them (cpython#125315).
warm_platform_cache()


@pytest.fixture(autouse=True)
def _restore_recursion_limit() -> Iterator[None]:
    """Undo a test's recursion-limit change (in-process `mypy.api.run` raises it to 2**14)."""
    limit = sys.getrecursionlimit()
    yield
    sys.setrecursionlimit(limit)
