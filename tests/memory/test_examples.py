"""Run the memory examples offline and check what they print."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _run(name: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def test_quickstart_recalls_across_chats_for_the_same_user_only(capsys):
    assert _run("memory_quickstart.py", capsys) == [
        "I will remember that.",
        "From earlier chats: Ann prefers aisle seats",
        "No preference saved.",
        "['Ann prefers aisle seats']",
    ]


def test_privacy_example_shows_retention_export_and_erasure(capsys):
    assert _run("memory_privacy.py", capsys) == [
        "['Vegan']",
        "[('Vegan', 'chat-2')]",
        "1",
        "[] 1",
    ]
