import runpy
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "executor.py"


def test_executor_example_runs_each_entry_point(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "returns #0 error tool-args-invalid",
        "[['returns #0'], ['shipping #0']]",
        "['topics #0 #0']",
        "['A', 'B']",
        "invoiced cus_1024 for 45.00 USD",
        "warranty #0",
    ]
