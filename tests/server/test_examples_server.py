"""The HTTP server example runs (and type-checks) as documented."""

from __future__ import annotations

import runpy
from pathlib import Path

import mypy.api
import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "serve_tools.py"


def test_serve_tools_demo(capsys: pytest.CaptureFixture[str]):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "200 18.0",
        "202 pending_approval",
        "200 pending",
        "200 invoiced cus_1024 for 49.50 USD",
        "403 this run belongs to another principal",
    ]


def test_serve_tools_builds_the_app():
    pytest.importorskip("starlette")
    pytest.importorskip("mcp")
    namespace = runpy.run_path(str(EXAMPLE))
    paths = [route.path for route in namespace["make_app"]().routes]
    assert paths == ["/mcp", "/readyz", "/{path:path}"]


def test_serve_tools_type_checks_strictly():
    out, err, status = mypy.api.run(["--strict", "--no-error-summary", str(EXAMPLE)])
    assert status == 0, out + err
