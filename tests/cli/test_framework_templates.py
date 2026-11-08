"""The framework templates: each generated project runs offline against a fake framework."""

from __future__ import annotations

import importlib.util
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_adapter.cli import main
from agent_adapter.cli.scaffold import TEMPLATES, write_project
from tests.cli.framework_fakes import FAKES, write_fake

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORKS = sorted(FAKES)
TOOLS = "['shipping_rate', 'refund_order']"
# The real framework each template imports, for the contract tests.
REAL = {
    "pydantic-ai": "pydantic_ai",
    "agno": "agno",
    "strands": "strands",
    "crewai": "crewai",
    "adk": "google.adk",
}


def _project(name: str, tmp_path: Path) -> Path:
    """Generate template `name` into tmp_path/proj with its fake framework beside it."""
    project = tmp_path / "proj"
    assert main(["init", name, str(project)], io.StringIO()) == 0
    write_fake(name, tmp_path / "fakes")
    return project


def _python(project: Path, tmp_path: Path, *args: str, **env: str) -> str:
    """Run python in `project` with the fake framework first on the path; return stdout."""
    environ = {k: v for k, v in os.environ.items() if k != "AGENT_MODEL"}
    environ["PYTHONPATH"] = os.pathsep.join([str(tmp_path / "fakes"), str(ROOT)])
    done = subprocess.run(
        [sys.executable, *args],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=120,
        env={**environ, **env},
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_every_framework_template_has_a_fake():
    assert set(FAKES) == set(REAL) <= set(TEMPLATES)


@pytest.mark.parametrize("name", FRAMEWORKS)
def test_template_runs_offline_and_records_the_run(name, tmp_path):
    project = _project(name, tmp_path)
    template = TEMPLATES[name]
    out = _python(project, tmp_path, template.filename).splitlines()
    noun = "crew" if name == "crewai" else "agent"
    assert out == [
        f"tools: {TOOLS} - set AGENT_MODEL to run the {noun}",
        "refunded 20.00 USD on A1",  # the review approved it
    ]
    runs = list((project / ".agent_adapter" / "runs").rglob("*.jsonl"))
    assert runs, "the recorder wrote no run"
    assert "refund_order" in "".join(p.read_text(encoding="utf-8") for p in runs)


@pytest.mark.parametrize("name", FRAMEWORKS)
def test_template_runs_the_agent_when_a_model_is_set(name, tmp_path):
    project = _project(name, tmp_path)
    out = _python(project, tmp_path, TEMPLATES[name].filename, AGENT_MODEL="house-model")
    expected = "house-model answered with shipping_rate,refund_order"
    assert out == (f"{expected} (1 hook)" if name == "strands" else expected)


DENY = """
import asyncio
from agent_adapter.errors import ApprovalDenied
import {module} as app
try:
    asyncio.run(app.executor.run("refund_order", {{"order_id": "A1", "amount": 90.0}}))
except ApprovalDenied:
    print("denied")
"""


@pytest.mark.parametrize("name", FRAMEWORKS)
def test_template_review_denies_a_large_refund(name, tmp_path):
    project = _project(name, tmp_path)
    module = Path(TEMPLATES[name].filename).stem
    assert _python(project, tmp_path, "-c", DENY.format(module=module)) == "denied"


def test_adk_template_imports_without_google_genai(tmp_path):
    project = _project("adk", tmp_path)
    shutil.rmtree(tmp_path / "fakes" / "google" / "genai")  # the offline path must not need it
    code = "import sys, agent; print('google.genai' in sys.modules)"
    assert _python(project, tmp_path, "-c", code).splitlines()[-1] == "False"


@pytest.mark.parametrize("name", FRAMEWORKS)
def test_template_requirements_name_the_framework_extra(name, tmp_path):
    write_project(name, tmp_path)
    extra = "pydantic-ai" if name == "pydantic-ai" else name
    assert (
        (tmp_path / "requirements.txt")
        .read_text(encoding="utf-8")
        .startswith(f"agent-adapter[{extra}]~=")
    )


@pytest.mark.contract
@pytest.mark.parametrize("name", FRAMEWORKS)
def test_template_exports_real_framework_tools(name, tmp_path, monkeypatch):
    pytest.importorskip(REAL[name])
    monkeypatch.chdir(tmp_path)  # the recorder writes under the working folder
    write_project(name, tmp_path)
    path = tmp_path / TEMPLATES[name].filename
    spec = importlib.util.spec_from_file_location(f"generated_{name.replace('-', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        assert len(module.tools) == 2
    finally:
        module.record(module.executor).stop()
