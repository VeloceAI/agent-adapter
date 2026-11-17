import runpy
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "voxcat_quality.py"


def test_the_quality_example_runs_offline(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "vox cat refunds four lakh fifty thousand rupees on eighteenth November twenty "
        "twenty-six at two thirty P M.",
        "Doctor Rao paid one thousand two hundred forty-nine dollars and ninety-nine cents, "
        "approximately three point five percent more, on November eighteenth, twenty twenty-six.",
        "This is Rahul from Acme Corp.",
        "[('Rahool', 'Rahul'), ('Acmy Corp', 'Acme Corp')]",
        "Did you say 4,500? Shall I run refund with amount 4500? Say yes to confirm.",
        "denied: no answer",
        "pay Priyanka",
        "['clipping']",
        "['click']",
    ]
