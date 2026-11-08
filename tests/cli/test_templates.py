"""`agent-adapter init`: each template is a tested example, and each generated project runs."""

from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from agent_adapter.cli import main
from agent_adapter.cli.scaffold import TEMPLATE_DIR, TEMPLATES, write_project
from agent_adapter.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _generate(name: str, folder: Path) -> Path:
    write_project(name, folder)
    return folder / TEMPLATES[name].filename


def _run(path: Path, **env: str) -> str:
    """Run a generated file as a script in its own folder, offline."""
    environ = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    environ["PYTHONPATH"] = str(ROOT)
    done = subprocess.run(
        [sys.executable, path.name],
        cwd=path.parent,
        capture_output=True,
        text=True,
        timeout=120,
        env={**environ, **env},
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"generated_{path.parent.name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_each_template_is_its_tested_example_byte_for_byte(name):
    example = EXAMPLES / f"template_{TEMPLATES[name].source}.py"
    assert TEMPLATES[name].text() == example.read_text(encoding="utf-8"), (
        f"copy {example} to {TEMPLATE_DIR / (TEMPLATES[name].source + '.py.tmpl')}"
    )


def test_every_template_example_is_offered():
    examples = {p.stem.removeprefix("template_") for p in EXAMPLES.glob("template_*.py")}
    packaged = {p.name.removesuffix(".py.tmpl") for p in TEMPLATE_DIR.glob("*.py.tmpl")}
    assert examples == packaged == {t.source for t in TEMPLATES.values()}


def test_init_writes_the_file_and_pinned_requirements(tmp_path):
    out = io.StringIO()
    assert main(["init", "langgraph", str(tmp_path / "proj")], out) == 0
    project = tmp_path / "proj"
    assert (project / "graph.py").read_text(encoding="utf-8") == TEMPLATES["langgraph"].text()
    assert (project / "requirements.txt").read_text(encoding="utf-8") == (
        "agent-adapter[langgraph]~=1.1\n"
    )
    assert "pip install -r requirements.txt\n  python graph.py" in out.getvalue()


def test_init_never_overwrites_without_force(tmp_path, capsys):
    (tmp_path / "main.py").write_text("mine", encoding="utf-8")
    assert main(["init", "script", str(tmp_path)], io.StringIO()) == 2
    assert "not overwriting" in capsys.readouterr().err
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == "mine"
    assert main(["init", "script", str(tmp_path), "--force"], io.StringIO()) == 0
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == TEMPLATES["script"].text()


def test_init_rejects_an_unknown_template(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["init", "django", str(tmp_path)])
    assert caught.value.code == 2
    assert "invalid choice: 'django'" in capsys.readouterr().err
    with pytest.raises(ConfigError, match="no template named 'django'"):
        write_project("django", tmp_path)


def _symlink(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:  # Windows without the symlink privilege
        pytest.skip(f"cannot create symlinks here ({type(exc).__name__})")


@pytest.mark.parametrize("dangling", [True, False])
@pytest.mark.parametrize("force", [False, True])
def test_init_never_writes_through_a_symlink(tmp_path, dangling, force):
    target = tmp_path / "elsewhere.py"
    if not dangling:
        target.write_text("theirs", encoding="utf-8")
    project = tmp_path / "proj"
    project.mkdir()
    _symlink(project / "main.py", target)
    with pytest.raises(ConfigError, match="symlink"):
        write_project("script", project, force=force)
    assert target.exists() is not dangling
    if not dangling:
        assert target.read_text(encoding="utf-8") == "theirs"
    assert not (project / "requirements.txt").exists()


def test_init_checks_for_links_even_where_none_can_be_made(tmp_path, monkeypatch):
    real = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self.name == "main.py" or real(self))
    with pytest.raises(ConfigError, match="symlink"):
        write_project("script", tmp_path, force=True)
    assert not (tmp_path / "main.py").exists()


def test_init_refuses_a_file_that_would_land_outside_the_folder(tmp_path, monkeypatch):
    from agent_adapter.cli import scaffold

    escape = scaffold.Template("script", "../escaped.py", (), "python x")
    monkeypatch.setitem(scaffold.TEMPLATES, "escape", escape)
    with pytest.raises(ConfigError, match="outside"):
        write_project("escape", tmp_path / "proj")
    assert not (tmp_path / "escaped.py").exists()


def test_init_reports_an_unwritable_folder(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert main(["init", "script", str(blocker / "sub")], io.StringIO()) == 2
    assert "agent-adapter init:" in capsys.readouterr().err


def test_the_module_entry_point_runs(tmp_path):
    done = subprocess.run(
        [sys.executable, "-m", "agent_adapter.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("agent-adapter ")


# --- each generated project runs offline ---------------------------------------------------


def test_script_template_runs(tmp_path):
    assert _run(_generate("script", tmp_path)) == "18.0"


def test_fastapi_template_serves_tools(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    app = _load(_generate("fastapi", tmp_path)).app
    with TestClient(app) as client:
        ok = client.post("/tools/shipping_rate", json={"region": "us"})
        assert (ok.status_code, ok.json()) == (200, {"result": 18.0})
        missing = client.post("/tools/nope", json={})
        assert (missing.status_code, missing.json()["detail"]["kind"]) == (404, "tool-not-found")
        bad = client.post("/tools/shipping_rate", json={"region": 5})
        assert (bad.status_code, bad.json()["detail"]["kind"]) == (422, "tool-args-invalid")


def test_flask_template_serves_tools(tmp_path):
    pytest.importorskip("flask")
    client = _load(_generate("flask", tmp_path)).app.test_client()
    ok = client.post("/tools/shipping_rate", json={"region": "us"})
    assert (ok.status_code, ok.get_json()) == (200, {"result": 18.0})
    missing = client.post("/tools/nope", json={})
    assert (missing.status_code, missing.get_json()["kind"]) == (404, "tool-not-found")


def test_celery_template_runs_the_task_without_a_worker(tmp_path):
    pytest.importorskip("celery")
    assert _run(_generate("celery", tmp_path)) == "18.0"


def test_langgraph_template_runs_the_graph(tmp_path):
    pytest.importorskip("langgraph")
    assert _run(_generate("langgraph", tmp_path)) == "18.0"


class _SdkContext:
    """The one field the bridge reads from the SDK's ToolContext."""

    tool_call_id = "c1"


async def test_openai_agents_template_attaches_runnable_tools(tmp_path):
    pytest.importorskip("agents")
    path = _generate("openai-agents", tmp_path)
    assert _run(path) == "tools: ['shipping_rate'] - set OPENAI_API_KEY to run"
    (tool,) = _load(path).agent.tools
    output: Any = await tool.on_invoke_tool(_SdkContext(), '{"region": "us"}')
    assert "18.0" in str(output)


async def test_mcp_server_template_serves_tools(tmp_path):
    pytest.importorskip("mcp")
    from mcp import Client

    server = _load(_generate("mcp-server", tmp_path)).server
    async with Client(server) as client:
        assert [t.name for t in (await client.list_tools()).tools] == ["shipping_rate"]
        rate = await client.call_tool("shipping_rate", {"region": "us"})
        assert rate.structured_content == {"result": 18.0}
