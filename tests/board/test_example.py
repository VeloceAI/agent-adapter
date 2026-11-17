"""The board examples run offline: claims stop a repeat, a handoff reaches the next agent."""

import runpy
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "parallel_research.py"


def test_parallel_research_example(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    *lines, tokens = capsys.readouterr().out.splitlines()
    assert lines == [
        "Report: prices: up 4% this year; weather: sunny, 21 C",
        "searches run: 2 for 3 asks (a claim stopped the repeat)",
        "board entries: 12",
        "parallel branches: 3, slowest waited 0 ms: True",
        "contested claims: 1",
    ]
    assert tokens.startswith("input tokens: alpha ") and "writer" in tokens


def test_board_handoff_example(capsys):
    runpy.run_path(str(EXAMPLE.with_name("board_handoff.py")), run_name="__main__")
    answer, board, note = capsys.readouterr().out.splitlines()
    assert answer == "Plan for 2 days in Oslo, museums first, 300 EUR in all"
    assert board.startswith("board: 45 entries")
    assert note.endswith("(cap 600), 1 note") and int(note.split()[2]) <= 600
