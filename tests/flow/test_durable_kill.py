"""A worker process killed with `os._exit` mid-run; the parent process recovers the run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from tests.flow import durable_kill_worker as worker
from tests.work import ROOT

# imported by name, so its functions have the same identity (graph digest) as here
CHILD = "import sys; from tests.flow import durable_kill_worker as w; w.main(sys.argv[1])"


async def test_a_killed_process_recovers_in_another(tmp_path: Path, monkeypatch: Any) -> None:
    db, log = tmp_path / "runs.db", tmp_path / "calls.log"
    env = {**os.environ, "AA_KILL": "1", "AA_KILL_LOG": str(log), "PYTHONPATH": str(ROOT)}
    child = subprocess.run(
        [sys.executable, "-c", CHILD, str(db)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert child.returncode == 3
    monkeypatch.setattr(worker, "LOG", log)
    result = await worker.runner(str(db)).recover(worker.flow, "run-1")  # ship is idempotent
    assert (result.status, result.state.note) == ("done", "shipped o-1")
    assert log.read_text(encoding="utf-8").split() == ["charge", "ship", "ship"]
