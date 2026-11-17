import runpy
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "report_template.py"


def test_report_template_example_runs(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    out = capsys.readouterr().out
    assert "| Europe | 4.1M |" in out
    assert '  n1 -->|"30 days"| n2' in out
    assert "edited outlook" in out
    assert "Europe should keep growing in Q4." in out
