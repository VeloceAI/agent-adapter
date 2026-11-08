import runpy
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "multi_agent.py"


def test_multi_agent_example_runs_every_pattern(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "Supervisor: A child ticket is 9.0 EUR.",
        "['llm:agent.tickets', 'llm:agent.tickets', 'llm:tool_loop', 'llm:tool_loop', "
        "'tool:ticket_price', 'tool:tickets']",
        "Sunflowers: room 12, east wing.",
        "Sunflowers: room 12, east wing.",
        "A short poem about the east wing.",
    ]
