"""`agent-adapter migrate` through the main command line."""

from __future__ import annotations

import io
from pathlib import Path

from agent_adapter.cli import main

FIXTURES = Path(__file__).resolve().parent / "migrate_fixtures"


def test_main_runs_migrate_and_maps_a_missing_path_to_exit_2(tmp_path, capsys):
    out = io.StringIO()
    assert main(["migrate", str(FIXTURES.parent), "--diff"], out) == 0
    assert out.getvalue().startswith("agent-adapter migrate: read-only")
    assert main(["migrate", str(tmp_path / "missing")], io.StringIO()) == 2
    assert "not a file or folder" in capsys.readouterr().err
