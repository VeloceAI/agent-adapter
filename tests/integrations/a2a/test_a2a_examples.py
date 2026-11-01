"""The A2A examples run as documented."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytest.importorskip("a2a")
pytest.importorskip("starlette")

pytestmark = pytest.mark.contract

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"


def _run(name: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def test_a2a_server_lists_one_skill_per_tool(capsys: pytest.CaptureFixture[str]):
    assert _run("a2a_server.py", capsys) == ["['shipping_rate', 'send_invoice']"]


def test_a2a_import_calls_and_reports_the_remote_approval(capsys: pytest.CaptureFixture[str]):
    assert _run("a2a_import.py", capsys) == [
        "['shop_shipping_rate', 'shop_send_invoice']",
        "5.0",
        "input-required",
    ]
