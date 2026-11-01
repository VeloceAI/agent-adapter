from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

import agent_adapter.models._port as port_module

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def fixture(name: str) -> Any:
    """A recorded provider payload from tests/fixtures/<provider>/<file>.json."""
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _no_wmi(*_keys: str) -> Any:
    raise OSError("no WMI in tests")


@pytest.fixture
def wmi_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `platform._wmi_query` to swap out, also on Python 3.11 (which has no WMI path)."""
    if not hasattr(platform, "_wmi_query"):
        monkeypatch.setattr(platform, "_wmi_query", _no_wmi, raising=False)


@pytest.fixture
def warm_calls(monkeypatch: pytest.MonkeyPatch, wmi_query: None) -> list[dict[str, Any]]:
    """An unwarmed port on "Windows": each `platform.platform(**kw)` call lands here."""
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(port_module, "_platform_warmed", False)
    monkeypatch.setattr(platform, "platform", lambda **kw: seen.append(kw))
    return seen
