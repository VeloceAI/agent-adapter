"""The durable flow example: a crashed run finishes on a new worker, the card charged once."""

import runpy
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "durable_flow.py"


def test_durable_flow_recovers_without_charging_twice(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "worker 1 died while emailing the tickets",
        "in doubt: ['email_tickets']",
        "done | tickets for Night at the museum sent",
        "{'charge_card': 1, 'email_tickets': 2, 'hold_seats': 1}",
    ]
