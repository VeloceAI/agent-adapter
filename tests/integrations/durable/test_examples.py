"""The durable-engine examples: a refund approved and paid once, or a clean skip without the SDK."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
DONE = ["waiting for review", "done | refunded 40.00 EUR on t-17"]


def _run(name: str, capsys) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


@pytest.mark.skipif(importlib.util.find_spec("dbos") is not None, reason="dbos is installed")
def test_the_dbos_example_skips_without_dbos(capsys):
    assert _run("durable_dbos.py", capsys) == [
        'dbos is not installed: pip install "agent-adapter[dbos]"'
    ]


@pytest.mark.skipif(importlib.util.find_spec("temporalio") is not None, reason="installed")
def test_the_temporal_example_skips_without_temporalio(capsys):
    assert _run("durable_temporal.py", capsys) == [
        'temporalio is not installed: pip install "agent-adapter[temporal]"'
    ]


@pytest.mark.contract
def test_the_dbos_example_pays_after_the_approval(capsys):
    pytest.importorskip("dbos")
    assert _run("durable_dbos.py", capsys)[-2:] == DONE


@pytest.mark.contract
def test_the_temporal_example_pays_after_the_approval(capsys):
    pytest.importorskip("temporalio")
    lines = _run("durable_temporal.py", capsys)
    assert lines[-2:] == DONE
