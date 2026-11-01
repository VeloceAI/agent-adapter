import runpy
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_optimiser_demo_finds_the_lookup_rule_and_leaves_the_host_alone(capsys):
    runpy.run_path(str(EXAMPLES / "optimiser_demo.py"), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "chosen: agg1",
        "base mean 0.0 -> 1.0",
        "case runs 8, proposer calls 2",
        "--- a/prompt/system",
        "['harness.after.json', 'harness.patch', 'report.md', 'result.json', 'state.json']",
        "You answer customer-support questions.",
    ]
