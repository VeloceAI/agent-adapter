"""Shared setup for the golden-path tests: import the example package fresh, in a temp folder."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
PACKAGE = "golden_path"


def _forget() -> None:
    for name in [m for m in sys.modules if m == PACKAGE or m.startswith(PACKAGE + ".")]:
        del sys.modules[name]


@pytest.fixture
def golden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """`golden_path.agent`, imported fresh; its default desk lives under `tmp_path / "work"`."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)  # the default runs folder (.agent_adapter/runs) is relative
    monkeypatch.setenv("SUPPORT_DESK_DATA", str(work / ".support_desk"))
    monkeypatch.delenv("SUPPORT_DESK_KEY", raising=False)
    monkeypatch.syspath_prepend(str(EXAMPLES))
    _forget()
    import golden_path.agent as agent

    try:
        yield agent
    finally:
        for built in (agent.default_desk, agent.eval_desk):
            if built.cache_info().currsize:
                built().close()
        _forget()
