"""`agent-adapter audit verify|export`, through its own parser."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

from agent_adapter.audit._files import segments
from agent_adapter.cli.audit import add_audit_parser, run_audit
from agent_adapter.errors import ConfigError
from tests.audit.helpers import KEY, write_log


def run(*argv: str) -> tuple[int, str]:
    parser = argparse.ArgumentParser(prog="agent-adapter")
    add_audit_parser(parser.add_subparsers(dest="command"))
    out = io.StringIO()
    return run_audit(parser.parse_args(["audit", *argv]), out), out.getvalue()


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AA_AUDIT_KEY", KEY)


def test_verify_passes_a_clean_log(tmp_path: Path) -> None:
    write_log(tmp_path, 2)
    code, text = run("verify", str(tmp_path))
    assert code == 0 and text.startswith("intact: 1 segment(s), 2 event(s)")


def test_verify_fails_a_tampered_log_and_lists_problems(tmp_path: Path) -> None:
    write_log(tmp_path, 2)
    ((_, path),) = segments(tmp_path)
    path.write_bytes(path.read_bytes().replace(b"refund", b"refunx"))
    code, text = run("verify", str(tmp_path))
    assert code == 1 and "NOT intact" in text and "  " in text.splitlines()[1]


def test_export_writes_csv_to_stdout_or_a_file(tmp_path: Path) -> None:
    write_log(tmp_path / "log", 2)
    code, text = run("export", str(tmp_path / "log"), "--format", "csv", "--since", "2020-01-01")
    assert code == 0 and text.startswith("seq,time,action") and len(text.splitlines()) == 3
    target = tmp_path / "out.jsonl"
    code, text = run("export", str(tmp_path / "log"), "--out", str(target))
    assert (
        code == 0
        and text == f"wrote 2 record(s) to {target}\n"
        and len(target.read_text().splitlines()) == 2
    )
    with pytest.raises(ConfigError, match="--force"):
        run("export", str(tmp_path / "log"), "--out", str(target))
    assert run("export", str(tmp_path / "log"), "--out", str(target), "--force")[0] == 0


def test_the_key_comes_only_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_log(tmp_path, 1)
    monkeypatch.delenv("AA_AUDIT_KEY")
    with pytest.raises(ConfigError, match="AA_AUDIT_KEY"):
        run("verify", str(tmp_path))
    monkeypatch.setenv("SHOP_AUDIT_KEY", KEY)
    assert run("verify", str(tmp_path), "--key-env", "SHOP_AUDIT_KEY")[0] == 0


def test_a_failed_export_leaves_no_file(tmp_path: Path) -> None:
    write_log(tmp_path / "log", 2)
    ((_, path),) = segments(tmp_path / "log")
    path.write_bytes(path.read_bytes().replace(b"refund", b"refunx"))
    target = tmp_path / "out.csv"
    with pytest.raises(ConfigError, match="not intact"):
        run("export", str(tmp_path / "log"), "--format", "csv", "--out", str(target))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["log"]


def test_verify_prints_an_anchor_that_catches_a_rollback(tmp_path: Path) -> None:
    write_log(tmp_path, 2)
    code, text = run("verify", str(tmp_path))
    anchor = text.splitlines()[-1].removeprefix("anchor ")
    assert code == 0 and anchor.count(":") == 2
    assert run("verify", str(tmp_path), "--anchor", anchor)[0] == 0
    stale = anchor.replace(anchor.split(":")[0], "log_other")
    assert run("verify", str(tmp_path), "--anchor", stale)[0] == 1


def test_export_never_overwrites_without_force(tmp_path: Path) -> None:
    write_log(tmp_path / "log", 1)
    target = tmp_path / "kept.csv"
    target.write_text("mine")
    with pytest.raises(ConfigError, match="--force"):
        run("export", str(tmp_path / "log"), "--out", str(target))
    assert target.read_text() == "mine"
