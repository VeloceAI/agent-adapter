"""The chat memory and approval memory examples run offline and print what the docs show."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples"


def run(name: str, *args: str, cwd: Path) -> list[str]:
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / name), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.splitlines()


def test_chat_memory_recalls_in_a_second_chat(tmp_path: Path) -> None:
    lines = run("chat_memory.py", "--check", cwd=tmp_path)
    assert lines == [
        "recalled: ['I prefer aisle seats']",
        "answer: Booked for you, keeping in mind: I prefer aisle seats.",
        "stored: 1 record(s) for ann",
    ]
    assert not any(tmp_path.iterdir()), "it writes only to a temporary folder"


def test_approval_memory_advises_proposes_and_still_asks(tmp_path: Path) -> None:
    lines = run("approval_memory.py", cwd=tmp_path)
    assert lines[0] == "pending - 5 similar: 5 approved, 0 denied (by ann, bob)"
    assert lines[1].startswith(
        "proposed: skip asking for 'refund' only when 10 <= 'amount' <= 40 and "
        "'reason' == 'damaged' (5 approved by 2, 0 against"
    )
    assert "confirm = confirm_from_config(" in "\n".join(lines)
    assert lines[-4:] == ["refunded 15 (damaged)"] + ["asks again: True"] * 3
