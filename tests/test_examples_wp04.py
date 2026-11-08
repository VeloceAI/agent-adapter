import importlib.util
import runpy
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _run(name: str, capsys) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def test_cost_prices_overrides_and_ledger(capsys):
    lines = _run("cost.py", capsys)
    assert lines == [
        "gpt-4.1",
        "0.341",
        "1.25",
        "4.0",
        "True None",
        "2026-11-05+acme-2026-11",
        "0.0325",
        "['plan', 'rerank'] 0",
        "3 2",
    ]


def test_tracing_sinks_and_otel(capsys):
    lines = _run("tracing.py", capsys)
    assert lines[:2] == [
        "['tool.started', 'tool.finished']",
        "['tool.started', 'custom.progress', 'tool.finished']",
    ]
    has_sdk = importlib.util.find_spec("opentelemetry") is not None and (
        importlib.util.find_spec("opentelemetry.sdk") is not None
    )
    if not has_sdk:
        assert lines[2] == "opentelemetry-sdk not installed"
    else:
        assert lines[2] == "execute_tool search search run_1"
