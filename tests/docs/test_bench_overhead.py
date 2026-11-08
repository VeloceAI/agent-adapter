"""The overhead benchmark runs (a few calls each) and prints its Markdown table."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bench_overhead.py"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_overhead", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_overhead"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules["bench_overhead"]
    return module


def test_the_benchmark_prints_one_row_per_scenario() -> None:
    bench = load()
    out = io.StringIO()
    assert bench.main(["--iterations", "3", "--warmup", "1"], out=out) == 0
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("agent_adapter ") and lines[0].endswith("; 3 calls each")
    assert lines[2] == "| Scenario | Median (us) | p95 (us) |"
    rows = [line for line in lines[4:] if line.startswith("| ")]
    assert len(rows) == len(bench.scenarios()) == 13
    for row in rows:
        median, p95 = (float(cell) for cell in row.strip("| ").split(" | ")[1:])
        assert 0 <= median <= p95


def test_every_scenario_gets_exactly_the_iterations_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = load()
    real = bench.sample
    counted: dict[int, int] = {}

    async def counting(step: object, calls: int) -> list[float]:
        counted[id(step)] = counted.get(id(step), 0) + calls
        samples: list[float] = await real(step, calls)
        return samples

    monkeypatch.setattr(bench, "sample", counting)
    rows = asyncio.run(bench.measure(bench.ROUND + 7, 0))
    assert len(rows) == 13
    assert set(counted.values()) == {bench.ROUND + 7}


def test_the_benchmark_refuses_zero_iterations(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        load().main(["--iterations", "0"])
    assert exc.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err


def test_production_logging_formats_info_records_and_restores_the_level() -> None:
    import logging

    bench = load()
    logger = logging.getLogger("agent_adapter")
    before = (logger.level, list(logger.handlers))
    with bench.production_logging():
        assert logger.isEnabledFor(logging.INFO)
        [handler] = [h for h in logger.handlers if h not in before[1]]
        assert handler.formatter is not None
    assert (logger.level, list(logger.handlers)) == before
