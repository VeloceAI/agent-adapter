"""examples/openapi_gateway.py: the 3.0.3 documents the docs show, run as written."""

import runpy
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "openapi_gateway.py"


def test_the_example_writes_union_free_3_0_documents(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "3.0.3 ['/tools/refund_status/invoke']",
        "0 0",
    ]


def test_the_optional_field_is_nullable_and_the_gateway_has_a_server() -> None:
    names = runpy.run_path(str(EXAMPLE))
    refund = names["generators"]["components"]["schemas"]["refund_status__Refund"]
    assert refund["properties"]["reason"]["nullable"] is True
    assert names["gateway"]["servers"] == [{"url": "https://tools.example.com"}]
    assert "security" not in names["gateway"]


def test_the_example_type_checks_strictly() -> None:
    mypy_api = pytest.importorskip("mypy.api")
    out, err, status = mypy_api.run(["--strict", "--no-error-summary", str(EXAMPLE)])
    assert status == 0, out + err
