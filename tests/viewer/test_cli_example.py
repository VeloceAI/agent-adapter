"""`agent-adapter ui` and the fan-out example."""

from __future__ import annotations

import asyncio
import io
import runpy
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.cli.main import main
from agent_adapter.exec.defaults import reset_default_executor
from agent_adapter.viewer import DEFAULT_FOLDER, server
from agent_adapter.viewer.server import ViewerServer
from tests.viewer.helpers import refreshed

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "viewer_fan_out.py"


def run_cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    return main(list(argv), out), out.getvalue()


def test_ui_export_writes_one_file(tmp_path: Path) -> None:
    code, text = run_cli("ui", str(tmp_path / "runs"), "--export", str(tmp_path / "r.html"))
    assert code == 0 and text.startswith("wrote ")
    assert (tmp_path / "r.html").is_file()


def test_ui_refuses_a_non_loopback_host(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = run_cli("ui", str(tmp_path), "--host", "0.0.0.0")
    assert code == 2
    assert "binds only to 127.0.0.1" in capsys.readouterr().err


def test_ui_approvals_need_a_reviewer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = run_cli("ui", str(tmp_path), "--approvals", str(tmp_path / "a.db"))
    assert code == 2
    assert "--reviewer" in capsys.readouterr().err


def test_ui_serves_until_interrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[ViewerServer] = []

    def interrupted(self: ViewerServer, poll_interval: float = 0.5) -> None:
        served.append(self)
        raise KeyboardInterrupt

    monkeypatch.setattr(server.ViewerServer, "serve_forever", interrupted)
    code, text = run_cli("ui", str(tmp_path), "--port", "0")
    assert code == 0
    assert text.startswith("agent-adapter ui: open http://127.0.0.1:")
    assert text.endswith("stopped\n")
    assert f"/#token={served[0].token}\n" in text and "keep it private" in text
    assert served and served[0].desk is None


def test_ui_is_listed_in_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "agent-adapter ui" in capsys.readouterr().out


def test_example_records_a_fan_out_run_and_prints_the_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module: dict[str, Any] = runpy.run_path(str(EXAMPLE))
    runs = asyncio.run(module["main"](tmp_path))
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "done 8 3"
    assert out[1] == "['ok', 'ok', 'ok']"
    assert out[2].startswith("pending approval apr_")
    assert out[3].startswith("agent-adapter ui ") and "--approvals" in out[3]
    folder = refreshed(runs)
    statuses = sorted(summary["status"] for summary in folder.summaries())
    assert statuses == ["ok", "ok", "paused"]
    (flow_run,) = [s for s in folder.summaries() if s["flows"]]
    assert flow_run["cost_usd"] and flow_run["framework"] == "example"
    detail = folder.detail(flow_run["key"])
    assert detail is not None
    spans = detail["spans"]
    searches = [s for s in spans if s["family"] == "tool" and not s["cached"]]
    assert len({s["lane"] for s in searches}) == 3  # the fan-out ran side by side
    assert sum(s["cached"] for s in spans if s["family"] == "tool") == 3  # the loop hit the cache
    assert any(s["family"] == "context" and s["tokens"] for s in spans)
    graph = detail["graphs"]["help_search"]
    kinds = {n["id"]: (n["fan_out"], n["fan_in"]) for n in graph["nodes"]}
    assert kinds["search"] == (True, False) and kinds["merge"] == (False, True)


def test_example_command_line_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "report.html"
    argv = [str(EXAMPLE), "--folder", str(tmp_path / "demo"), "--export", str(out)]
    monkeypatch.setattr("sys.argv", argv)
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert out.is_file() and "help_search" in out.read_text(encoding="utf-8")


def test_every_setup_records_to_the_default_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    reset_default_executor()
    try:
        runpy.run_path(str(ROOT / "examples" / "viewer_setups.py"), run_name="__main__")
    finally:
        reset_default_executor()
    assert capsys.readouterr().out.splitlines() == [
        "Hotel Central, Lisbon",
        "Hotel Central, Porto",
        "Hotel Central, Faro",
        "Hotel Central, Braga",
    ]
    folder = refreshed(tmp_path / DEFAULT_FOLDER)
    sources = sorted(summary["framework"] for summary in folder.summaries())
    assert sources == ["", "celery", "fastapi", "langgraph"]


@pytest.mark.parametrize("port", ["-1", "65536", "http"])
def test_ui_refuses_a_bad_port(port: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["ui", "--port", port])
    assert stopped.value.code == 2
    assert "port" in capsys.readouterr().err


def test_ui_export_refuses_approvals_and_run_needs_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, db = str(tmp_path / "r.html"), str(tmp_path / "a.db")
    code, _ = run_cli("ui", str(tmp_path), "--export", out, "--approvals", db)
    assert code == 2 and "drop --approvals" in capsys.readouterr().err
    code, _ = run_cli("ui", str(tmp_path), "--run", "run_1")
    assert code == 2 and "--run needs --export" in capsys.readouterr().err


def test_ui_reads_saved_evals_from_the_evals_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    served: list[ViewerServer] = []

    def interrupted(self: ViewerServer, poll_interval: float = 0.5) -> None:
        served.append(self)
        raise KeyboardInterrupt

    monkeypatch.setattr(server.ViewerServer, "serve_forever", interrupted)
    code, _ = run_cli("ui", str(tmp_path), "--port", "0", "--evals", str(tmp_path / "ev"))
    assert code == 0 and served[0].evals.folder == (tmp_path / "ev").resolve()
    code, _ = run_cli("ui", str(tmp_path), "--export", str(tmp_path / "r.html"), "--evals", "x")
    assert code == 2 and "drop --evals" in capsys.readouterr().err
