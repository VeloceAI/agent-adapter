"""Run the production-to-evals example offline and check the loop it shows."""

import runpy
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "evals_from_production.py"


def test_recorded_runs_become_cases_the_fix_passes(tmp_path: Path) -> None:
    lines = runpy.run_path(str(EXAMPLE), run_name="example")["main"](tmp_path)
    assert lines == [
        "run-otlp-t1: 'Where is order 1042?' -> 'Order 1042 shipped.' ['status:ok']",
        "run-otlp-t2: 'Cancel order 1043' -> 'cancelled' ['status:failed']",
        "old_agent: 1/2 passed",
        "fixed_agent: 2/2 passed",
    ]
    assert (tmp_path / "eval_cases.jsonl").is_file()
