"""The audit example runs offline, writes only to a temporary folder, prints what docs show."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_audit_example_verifies_exports_and_catches_a_change(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "audit_log.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == [
        "refund: approval-denied",
        "lookup: guardrail-blocked",
        "intact: 1 segment(s), 7 event(s), last segment sealed",
        "['approval.requested', 'approval.decided', 'approval.consumed', 'approval.requested', "
        "'approval.decided', 'approval.denied', 'guardrail.blocked']",
        "seq,time,action,run_id,call_id,data,prev,mac",
        "NOT intact (2 problem(s)): 1 segment(s), 7 event(s), last segment sealed",
    ]
    assert not any(tmp_path.iterdir())


def test_audit_ship_example_checks_the_received_stream(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "audit_ship.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == [
        "intact: 1 segment(s), 3 event(s), last segment sealed",
        "NOT intact (3 problem(s)): 1 segment(s), 2 event(s), last segment sealed",
    ]
    assert not any(tmp_path.iterdir())
