import runpy
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "report_template.py"


def test_report_template_example_runs(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    out = capsys.readouterr().out
    assert "| Europe | 4.1M | new partners [2] |" in out
    assert '  n1 -->|"30 days"| n2' in out
    assert "Europe should keep growing [2] in Q4." in out
    assert "2. [Europe desk notes](https://example.com/eu)" in out
    assert "editor: 1 calls" in out and "issues: none" in out
    assert "prompt sizes (chars): blank at most" in out
    assert "html True <h1>Acme &amp; Co</h1>" in out and "xml True" in out
    assert "footnote: Revenue grew 12% in Q3[^1], led by Europe." in out
    assert "custom: Revenue grew 12% in Q3<sup>1</sup>, led by Europe." in out
    assert "verifier: 2 checks, summary written 2 times, issues: none" in out
