"""`agent-adapter doctor`: each check, and the exit code."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pytest

from agent_adapter._version import __version__
from agent_adapter.cli import doctor, main, pipindex
from agent_adapter.integrations.base import IntegrationBase

REQUIRES = [
    "pydantic<3,>=2.12",
    'openai<4,>=3.41; extra == "openai"',
    'mcp<3,>=2.3; extra == "mcp"',
    'not-a-real-dist>=1; extra == "ghost"',
    'agent-adapter[mcp]; extra == "server"',
    'agent-adapter[ghost,mcp]; extra == "everything"',
    'agent-adapter[loop]; extra == "loop"',
]


class Fake(IntegrationBase):
    """A discovered integration with chosen requirements."""

    def __init__(self, name: str, requires: tuple[str, ...], extra: str) -> None:
        self.name = name  # type: ignore[misc]
        self.requires = requires  # type: ignore[misc]
        self.extra = extra  # type: ignore[misc]

    def export_tool(self, spec: Any, *, executor: Any = None) -> Any:
        return spec


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {"pydantic": "2.12.0", "openai": "3.41.0", "mcp": "2.3.0"}
    monkeypatch.setattr(doctor, "_version", versions.get)


def test_extras_table_parses_requirements_and_nested_extras():
    table = doctor.extras_table(REQUIRES)
    assert table["openai"] == [("openai", ())]
    assert table["everything"] == [("agent-adapter", ("ghost", "mcp"))]
    assert "pydantic" not in str(table)  # a core dependency belongs to no extra


def test_an_extra_is_installed_only_with_all_of_its_distributions(installed):
    table = doctor.extras_table(REQUIRES)
    assert doctor.extra_installed("openai", table)
    assert doctor.extra_installed("server", table)  # through agent-adapter[mcp]
    assert not doctor.extra_installed("ghost", table)
    assert not doctor.extra_installed("everything", table)
    assert doctor.extra_installed("loop", table)  # a self-reference ends, it does not recurse
    assert not doctor.extra_installed("unknown", table)
    assert doctor.check_extras(table).message == "loop, mcp, openai, server"


def test_python_older_than_3_11_is_an_error():
    assert doctor.check_python((3, 10)).level == "error"
    assert doctor.check_python((3, 12)).level in {"ok", "warn"}


def test_outside_a_virtual_environment_warns(monkeypatch):
    monkeypatch.setattr(doctor.sys, "base_prefix", doctor.sys.prefix)
    finding = doctor.check_python((3, 12))
    assert finding.level == "warn"
    assert "not in a virtual environment" in finding.message


def test_package_metadata_must_match_the_code():
    assert doctor.check_package(__version__).level == "ok"
    stale = doctor.check_package("0.0.1")
    assert stale.level == "warn"
    assert "reinstall" in stale.message
    assert "is not installed" in doctor.check_package(None).message


def test_integrations_show_versions_extras_and_skipped_plugins(installed, monkeypatch):
    table = doctor.extras_table(REQUIRES)
    found = {
        "mcp": Fake("mcp", ("json",), "mcp"),
        "ghost": Fake("ghost", ("not_a_real_module",), "ghost"),
        "bare": Fake("bare", ("not_a_real_module",), ""),
    }
    discovery = logging.getLogger("agent_adapter.integrations.discovery")
    handlers = list(discovery.handlers)

    def discover(*, refresh: bool = False) -> dict[str, Any]:
        discovery.warning("integration %r failed to load from %s; skipped", "broken", "x:y")
        return found

    monkeypatch.setattr(doctor, "available_integrations", discover)
    lines = [f.line() for f in doctor.check_integrations(table)]
    assert lines == [
        "[warn ] integrations: integration 'broken' failed to load from x:y; skipped",
        "[info ] integration bare: integration 'bare' needs not_a_real_module: "
        "install not_a_real_module",
        '[info ] integration ghost: pip install "agent-adapter[ghost]"',
        "[ok   ] integration mcp: ready mcp 2.3.0",
    ]
    assert discovery.propagate is True
    assert discovery.handlers == handlers  # restored


def test_a_local_file_named_like_an_sdk_is_flagged(tmp_path):
    (tmp_path / "mcp.py").write_text("", encoding="utf-8")
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "__init__.py").write_text("", encoding="utf-8")
    found = {"mcp": Fake("mcp", ("mcp",), "mcp"), "oa": Fake("oa", ("agents",), "x")}
    flagged = [f.message for f in doctor.check_shadowing(tmp_path, found)]
    assert len(flagged) == 2
    assert all("rename it" in message for message in flagged)


def test_the_checkout_itself_is_not_shadowing():
    root = Path(doctor.__file__).resolve().parents[2]
    assert doctor.check_shadowing(root, {}) == []


def test_doctor_exits_1_on_an_error_and_0_otherwise(monkeypatch):
    monkeypatch.setattr(pipindex, "config_files", lambda environ: [])
    monkeypatch.setenv("PIP_INDEX_URL", "https://pkgs.example.com/simple")
    monkeypatch.delenv("PIP_EXTRA_INDEX_URL", raising=False)
    monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
    out = io.StringIO()
    assert main(["doctor"], out) == 0
    text = out.getvalue()
    assert "[ok   ] pip index: https://pkgs.example.com/simple (from $PIP_INDEX_URL)" in text
    assert "integration openai: ready" in text
    too_old = doctor.check_python((3, 10))
    monkeypatch.setattr(doctor, "check_python", lambda: too_old)
    out = io.StringIO()
    assert main(["doctor"], out) == 1
    assert out.getvalue().splitlines()[-1].startswith("1 error(s)")


def test_doctor_without_installed_metadata(monkeypatch):
    def absent(name: str) -> Any:
        raise doctor.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(doctor.importlib.metadata, "distribution", absent)
    monkeypatch.setattr(pipindex, "config_files", lambda environ: [])
    package = doctor.diagnose({}, Path.cwd())[1]
    assert package.level == "warn"
    assert "is not installed" in package.message
