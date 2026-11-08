"""Suite-wide setup."""

import gc
import sys
from collections.abc import Iterator

import pytest

from agent_adapter.models._port import warm_platform_cache
from tests import sqlite_guard

# Fill platform's caches without WMI before any test or SDK queries them (cpython#125315).
warm_platform_cache()


@pytest.fixture(autouse=True)
def _restore_recursion_limit() -> Iterator[None]:
    """Undo a test's recursion-limit change (in-process `mypy.api.run` raises it to 2**14)."""
    limit = sys.getrecursionlimit()
    yield
    sys.setrecursionlimit(limit)


sqlite_guard.install()


@pytest.fixture(autouse=True)
def _no_leaked_sqlite_connections(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail the test during which an unclosed sqlite3 connection was freed."""
    sqlite_guard.enter(request.node.nodeid)
    yield
    leaks = sqlite_guard.take_leaks()
    if leaks:
        pytest.fail("sqlite3 connection freed without close():\n" + "\n".join(leaks))


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Fail the run when a connection left open by some test is only freed at the end."""
    gc.collect()
    leaks = sqlite_guard.take_leaks()
    if leaks:
        session.shouldfail = "sqlite3 connections freed without close():\n" + "\n".join(leaks)
