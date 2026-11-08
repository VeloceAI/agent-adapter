import runpy
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "guardrails.py"


def test_guardrails_example_blocks_redacts_and_asks(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "{'name': 'Ana', 'email': '[EMAIL]', 'phone': '[PHONE]'}",
        "sent 'June'",
        "approval asked for send_newsletter",
        "sent 'June'",
        "guardrail-blocked | guardrail blocked args for tool 'save_note': found card",
        "Ana's address is [EMAIL].",
        "guardrail-blocked | guardrail blocked args for tool 'save_note': resale is not allowed",
    ]
