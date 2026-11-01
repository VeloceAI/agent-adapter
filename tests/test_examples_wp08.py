import runpy
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_tool_loop_example(capsys):
    runpy.run_path(str(EXAMPLES / "tool_loop.py"), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "stop 2 The exchange_rate result is 0.92.",
        "0.0048",
        "True 0.0048",
        "exchange_rate llm:answer:0 False",
        "pending",
        "stop The send_invoice result is invoiced cus_1024 for 49.50 USD.",
        "2 5",
    ]
