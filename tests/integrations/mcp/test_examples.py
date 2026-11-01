"""The MCP examples run (and type-check) as documented."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import mypy.api

pytestmark = pytest.mark.contract

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
FILES = ("mcp_server.py", "mcp_import.py", "mcp_http.py")


def _run(name: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def test_mcp_server_demo_lists_calls_and_denies(capsys: pytest.CaptureFixture[str]):
    lines = _run("mcp_server.py", capsys)
    assert lines[0].startswith("shipping_rate ") and "read_only_hint=True" in lines[0]
    assert lines[1].startswith("send_invoice ") and "destructive_hint=True" in lines[1]
    assert lines[2:] == [
        "{'result': 18.0}",
        "True approval denied for tool 'send_invoice': no approval port configured",
    ]


def test_mcp_import_resumes_the_remote_approval(capsys: pytest.CaptureFixture[str]):
    assert _run("mcp_import.py", capsys) == [
        "['shop_shipping_rate', 'shop_send_invoice']",
        "read_only",
        "5.0",
        "pending True",
        "invoiced cus_1024 for 49.50 USD",
    ]


def test_mcp_http_builds_the_app_at_mcp(capsys: pytest.CaptureFixture[str]):
    assert _run("mcp_http.py", capsys) == ["['/mcp']"]


def test_mcp_examples_type_check_strictly():
    files = [str(EXAMPLES / name) for name in FILES]
    out, err, status = mypy.api.run(["--strict", "--no-error-summary", *files])
    assert status == 0, out + err
