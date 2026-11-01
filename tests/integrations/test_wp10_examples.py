"""The LangChain, Gemini and OpenAPI examples run (each skips without its extra)."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _run(name: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def test_langchain_example_exports_and_imports(capsys):
    pytest.importorskip("langchain_core")
    lines = _run("langchain_tools.py", capsys)
    assert lines == ['{"region": "us", "rate": 18.0, "call_id": "call_1"}', "lc_word_count 3"]


def test_gemini_example_answers_a_function_call(capsys, monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert _run("gemini_afc.py", capsys) == ["shipping_rate {'result': 18.0}"]


def test_openapi_example_imports_and_calls_an_operation(capsys):
    pytest.importorskip("httpx")
    pytest.importorskip("yaml")
    assert _run("openapi_import.py", capsys) == [
        "['getRate'] ['item', 'X_Region']",
        "{'item': 'laptop', 'region': 'EU'}",
    ]


def test_wp10_examples_type_check_strictly():
    for root in ("langchain_core", "google.genai", "httpx", "yaml"):
        pytest.importorskip(root)
    import mypy.api

    names = ("langchain_tools.py", "gemini_afc.py", "openapi_import.py")
    out, err, status = mypy.api.run(
        ["--strict", "--no-error-summary", *(str(EXAMPLES / n) for n in names)]
    )
    assert status == 0, out + err
