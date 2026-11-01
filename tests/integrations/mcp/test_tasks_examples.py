"""The MCP tasks and OAuth examples run (and type-check) as documented."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import mypy.api

pytestmark = pytest.mark.contract

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
FILES = ("mcp_tasks.py", "mcp_oauth.py")


def _run(name: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def test_mcp_tasks_resumes_the_task_after_review(capsys: pytest.CaptureFixture[str]):
    assert _run("mcp_tasks.py", capsys) == [
        "pending task_pending True",
        "invoiced cus_1024 for 49.50 USD",
    ]


def test_mcp_oauth_builds_a_cimd_client(capsys: pytest.CaptureFixture[str]):
    assert _run("mcp_oauth.py", capsys) == [
        '"https://agent.example.com/oauth/client.json"',
        "OAuthClientProvider True",
    ]


def test_tasks_examples_type_check_strictly():
    files = [str(EXAMPLES / name) for name in FILES]
    out, err, status = mypy.api.run(["--strict", "--no-error-summary", *files])
    assert status == 0, out + err
