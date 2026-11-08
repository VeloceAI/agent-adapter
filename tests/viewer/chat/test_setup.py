"""Targets, `build_host` options, `serve(...)` and the `agent-adapter ui` chat flags."""

from __future__ import annotations

import importlib
import io
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.exec import Executor
from agent_adapter.viewer import serve, server
from agent_adapter.viewer.chat.setup import build_host
from agent_adapter.viewer.chat.targets import load_target
from agent_adapter.viewer.server import ViewerServer
from tests.fakes.models import ScriptedModel, text
from tests.viewer.chat.helpers import tools
from tests.viewer.chat.test_contract import async_text
from tests.viewer.test_cli_example import run_cli

HERE = "tests.viewer.chat.test_setup"
model = ScriptedModel([text("hi")])
registry = tools
executor = Executor(tools)
not_a_registry = "nope"


def test_a_target_loads_a_module_attribute() -> None:
    assert load_target(f"{HERE}:model", "--model") is model


@pytest.mark.parametrize(
    ("target", "why"),
    [
        ("tests/viewer/chat/test_setup.py:model", "module.path:attribute"),
        ("tests.viewer.chat.test_setup", "module.path:attribute"),
        ("x:y:z", "module.path:attribute"),
        ("", "module.path:attribute"),
        ("no_such_module_here:x", "cannot import no_such_module_here"),
        (f"{HERE}:missing", "has no 'missing'"),
    ],
)
def test_a_bad_target_is_a_config_error_naming_it(target: str, why: str) -> None:
    with pytest.raises(ConfigError, match="--agent") as found:
        load_target(target, "--agent")
    assert why in str(found.value)


@pytest.mark.parametrize(
    ("options", "why"),
    [
        ({}, "either an agent or a model"),
        ({"agent": async_text, "model": model}, "not both"),
        ({"agent": "not callable"}, "callable"),
        ({"model": "not a model"}, "LLMPort"),
        ({"agent": async_text, "executor": "x"}, "Executor"),
        ({"model": model, "registry": "x"}, "Registry"),
        ({"model": model, "max_steps": 0}, "max_steps"),
        ({"model": model, "turn_timeout": 0}, "turn_timeout"),
        ({"model": model, "approval_timeout": float("nan")}, "approval_timeout"),
        ({"model": model, "reviewer": " "}, "reviewer"),
    ],
)
def test_build_host_refuses_bad_options(tmp_path: Path, options: dict[str, Any], why: str) -> None:
    with pytest.raises(ConfigError, match=why):
        build_host(runs=tmp_path / "runs", chat_dir=tmp_path / "chats", **options)


def test_build_host_needs_a_runs_folder_not_a_file(tmp_path: Path) -> None:
    file = tmp_path / "run.jsonl"
    file.write_text("")
    with pytest.raises(ConfigError, match="runs folder"):
        build_host(runs=file, agent=async_text, chat_dir=tmp_path / "chats")


def no_serving(monkeypatch: pytest.MonkeyPatch) -> list[ViewerServer]:
    served: list[ViewerServer] = []

    def interrupted(self: ViewerServer, poll_interval: float = 0.5) -> None:
        served.append(self)
        raise KeyboardInterrupt

    monkeypatch.setattr(server.ViewerServer, "serve_forever", interrupted)
    return served


def test_serve_prints_the_link_opens_the_browser_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    served = no_serving(monkeypatch)
    opened: list[str] = []

    def fake_open(url: str) -> bool:
        opened.append(url)
        return False

    monkeypatch.setattr("webbrowser.open", fake_open)
    out = io.StringIO()
    with caplog.at_level(logging.WARNING):
        serve(
            async_text,
            folder=tmp_path / "runs",
            chat_dir=tmp_path / "chats",
            port=0,
            out=out,
        )
    lines = out.getvalue().splitlines()
    assert lines[0] == f"agent-adapter ui: open {served[0].url}"
    (launch,) = opened
    assert launch.startswith(f"{served[0].base_url}launch/") and served[0].token not in launch
    assert "chat is on" in lines[2] and lines[-1] == "stopped"
    assert "chat is on" in caplog.text and served[0].chat is not None
    assert served[0].chat.closed


def test_serve_needs_an_agent_or_a_model() -> None:
    with pytest.raises(ConfigError, match="agent= or model="):
        serve()


def test_a_taken_port_closes_the_chat_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    serve_module = importlib.import_module("agent_adapter.viewer.serving")

    def taken(*args: object, **kwargs: object) -> None:
        raise OSError("address in use")

    monkeypatch.setattr(serve_module, "make_server", taken)
    with pytest.raises(OSError):
        serve(async_text, folder=tmp_path / "runs", chat_dir=tmp_path / "chats", out=io.StringIO())


def test_ui_chat_flags_start_chat_from_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = no_serving(monkeypatch)
    code, out = run_cli(
        "ui",
        str(tmp_path / "runs"),
        "--port",
        "0",
        "--model",
        f"{HERE}:model",
        "--tools",
        f"{HERE}:registry",
        "--executor",
        f"{HERE}:executor",
        "--system",
        "be brief",
        "--max-steps",
        "3",
        "--turn-timeout",
        "60",
        "--approval-timeout",
        "30",
        "--reviewer",
        "ana",
        "--chat-dir",
        str(tmp_path / "chats"),
    )
    assert code == 0 and "chat is on" in out
    host = served[0].chat
    assert host is not None and host.mode == "builtin" and host.options.reviewer == "ana"
    assert host.options.turn_timeout == 60.0 and host.desk.timeout == 30.0
    assert (tmp_path / "chats").is_dir()


def test_ui_without_chat_saves_run_cases_in_chat_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = no_serving(monkeypatch)
    code, _ = run_cli(
        "ui", str(tmp_path / "runs"), "--port", "0", "--chat-dir", str(tmp_path / "c")
    )
    assert code == 0 and served[0].chat is None
    assert served[0].cases.path == tmp_path / "c" / "eval_cases.jsonl"


def test_ui_agent_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    served = no_serving(monkeypatch)
    monkeypatch.chdir(tmp_path)
    code, _ = run_cli("ui", "--port", "0", "--agent", "tests.viewer.chat.test_contract:async_text")
    assert code == 0
    host = served[0].chat
    assert host is not None and host.mode == "agent"
    assert host.options.agent_name == "tests.viewer.chat.test_contract:async_text"
    assert host.options.reviewer == "local"
    assert (tmp_path / ".agent_adapter" / "chats").is_dir()


@pytest.mark.parametrize(
    ("argv", "why"),
    [
        (["--system", "x"], "--system is a chat option"),
        (["--tools", f"{HERE}:registry"], "--tools is a chat option"),
        (["--model", f"{HERE}:model", "--agent", f"{HERE}:model"], "not both"),
        (
            ["--agent", "tests.viewer.chat.test_contract:async_text", "--max-steps", "2"],
            "--max-steps goes with --model",
        ),
        (["--model", f"{HERE}:model", "--tools", f"{HERE}:not_a_registry"], "Registry"),
        (["--model", "bad target"], "--model 'bad target'"),
        (["--model", f"{HERE}:model", "--export", "r.html"], "drop --model"),
    ],
)
def test_ui_refuses_bad_chat_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str], why: str
) -> None:
    code, _ = run_cli("ui", str(tmp_path / "runs"), "--port", "0", *argv)
    assert code == 2 and why in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "inf", "soon"])
def test_ui_refuses_bad_timeouts(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    from agent_adapter.cli.main import main

    with pytest.raises(SystemExit) as stopped:
        main(["ui", "--turn-timeout", value])
    assert stopped.value.code == 2 and "turn-timeout" in capsys.readouterr().err


def test_ui_help_shows_the_chat_examples(capsys: pytest.CaptureFixture[str]) -> None:
    from agent_adapter.cli.main import main

    with pytest.raises(SystemExit):
        main(["ui", "--help"])
    text = capsys.readouterr().out
    assert "agent-adapter ui --agent app:chat" in text and "--approval-timeout" in text


@pytest.mark.parametrize("value", ["0", "two"])
def test_ui_refuses_a_bad_max_steps(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    from agent_adapter.cli.main import main

    with pytest.raises(SystemExit) as stopped:
        main(["ui", "--max-steps", value])
    assert stopped.value.code == 2 and "max-steps" in capsys.readouterr().err


def test_ui_targets_import_from_the_current_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = no_serving(monkeypatch)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", str(tmp_path))])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chat_app_here.py").write_text(
        "async def chat(message, session):\n    return 'hi'\n", encoding="utf-8"
    )
    monkeypatch.delitem(sys.modules, "chat_app_here", raising=False)
    code, _ = run_cli("ui", "--port", "0", "--agent", "chat_app_here:chat")
    assert code == 0 and served[0].chat is not None
    assert sys.path[-1] == str(tmp_path)  # appended: never shadows installed modules
