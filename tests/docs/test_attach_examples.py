"""The attach() example and the CLI page run offline and match what they claim."""

import re
import runpy
from pathlib import Path

import pytest

from agent_adapter.cli.scaffold import TEMPLATES

ROOT = Path(__file__).resolve().parents[2]


def test_attach_example_attaches_and_shows_the_two_first_errors(capsys):
    runpy.run_path(str(ROOT / "examples" / "attach_any.py"), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "shipping_rate",
        "no integration named 'openai-agent'; did you mean 'openai_agents'?",
        "attach does not take approval",
    ]


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_cli_page_lists_every_template_with_its_file_and_run_command(name):
    page = (ROOT / "docs" / "reference" / "cli.md").read_text(encoding="utf-8")
    template = TEMPLATES[name]
    row = re.search(rf"^\| `{re.escape(name)}` \|.*$", page, re.M)
    assert row, name
    assert f"`{template.filename}`" in row.group(0)
    assert f"`{template.run}`" in row.group(0)
    assert f"examples/template_{template.source}.py" in row.group(0)
