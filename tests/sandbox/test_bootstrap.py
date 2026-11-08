import base64
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_adapter.sandbox import _bootstrap as boot

PY = [sys.executable, "-I", "-c"]


def env(tmp_path: Path) -> dict[str, str]:
    return boot.child_env({}, str(tmp_path))


def test_execute_captures_output_and_exit_code(tmp_path: Path) -> None:
    code = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(4)"
    outcome = boot.execute([*PY, code], str(tmp_path), env(tmp_path), 20, 1000)
    assert outcome["exit_code"] == 4 and not outcome["timed_out"]
    assert outcome["stdout"]["data"].strip() == b"out"
    assert outcome["stderr"]["data"].strip() == b"err"


def test_execute_caps_each_stream_and_keeps_draining(tmp_path: Path) -> None:
    code = "import sys; sys.stdout.write('x' * 300000); sys.stderr.write('y' * 10)"
    outcome = boot.execute([*PY, code], str(tmp_path), env(tmp_path), 20, 100)
    assert outcome["stdout"] == {"data": b"x" * 100, "truncated": True}
    assert outcome["stderr"] == {"data": b"y" * 10, "truncated": False}
    assert outcome["exit_code"] == 0


def test_execute_kills_on_timeout(tmp_path: Path) -> None:
    code = "import time; print('started', flush=True); time.sleep(30)"
    outcome = boot.execute([*PY, code], str(tmp_path), env(tmp_path), 1, 1000)
    assert outcome["timed_out"] and outcome["exit_code"] == -1
    assert outcome["stdout"]["data"].strip() == b"started"


def test_execute_reports_a_missing_command(tmp_path: Path) -> None:
    outcome = boot.execute(["no-such-command-xyz"], str(tmp_path), {}, 5, 1000)
    assert outcome["exit_code"] == 127
    assert b"cannot start no-such-command-xyz" in outcome["stderr"]["data"]


def test_child_env_holds_only_basics_and_the_given_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "abc")
    made = boot.child_env({"A": "1"}, str(tmp_path))
    assert "SECRET_TOKEN" not in made
    assert made["A"] == "1" and made["HOME"] == str(tmp_path)
    assert made["PYTHONIOENCODING"] == "utf-8"


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("a/b.txt", True),
        ("", False),
        ("../x", False),
        ("a\\b", False),
        ("C:x", False),
        ("a/./b", False),
        (3, False),
        ("a\x00", False),
    ],
)
def test_safe_name(name: object, ok: bool) -> None:
    assert boot.safe_name(name) is ok


def test_collect_returns_new_and_changed_files_only(tmp_path: Path) -> None:
    inputs = boot.write_inputs(
        str(tmp_path), "main.py", "print(1)", {"same.txt": _b64(b"s"), "d/edit.txt": _b64(b"e")}
    )
    (tmp_path / "d" / "edit.txt").write_bytes(b"changed")
    (tmp_path / "new.txt").write_bytes(b"new")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"c")
    found, cut = boot.collect(str(tmp_path), "main.py", inputs, 10, 1000)
    assert [(f["name"], f["size"], base64.b64decode(f["data"])) for f in found] == [
        ("new.txt", 3, b"new"),
        ("d/edit.txt", 7, b"changed"),
    ]
    assert not cut


def test_collect_bounds_files_and_bytes(tmp_path: Path) -> None:
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_bytes(b"abcd")
    found, cut = boot.collect(str(tmp_path), "main.py", {}, 2, 6)
    assert [(f["name"], base64.b64decode(f["data"])) for f in found] == [
        ("f0.txt", b"abcd"),
        ("f1.txt", b"ab"),
    ]
    assert cut
    found, cut = boot.collect(str(tmp_path), "main.py", {}, 5, 100)
    assert len(found) == 3 and not cut
    _, cut = boot.collect(str(tmp_path), "main.py", {}, 5, 5)
    assert cut


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_collect_skips_links(tmp_path: Path) -> None:  # pragma: no cover - POSIX only
    (tmp_path / "real.txt").write_bytes(b"r")
    os.symlink(tmp_path / "real.txt", tmp_path / "link.txt")
    found, _ = boot.collect(str(tmp_path), "main.py", {}, 5, 100)
    assert [f["name"] for f in found] == ["real.txt"]


def test_collect_stops_after_max_walk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boot, "MAX_WALK", 1)
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_bytes(b"a")
    found, cut = boot.collect(str(tmp_path), "main.py", {}, 10, 100)
    assert len(found) == 1 and cut


def _payload(tmp_path: Path, **changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "v": boot.PROTOCOL,
        "workdir": str(tmp_path),
        "code_file": "main.py",
        "code": "print(open('in.txt').read()); open('out.txt', 'w').write('o')",
        "command": None,
        "files": {"in.txt": _b64(b"hello")},
        "env": {},
        "timeout_s": 20,
        "max_output": 1000,
        "max_files": 4,
        "max_file_bytes": 1000,
    }
    body.update(changes)
    return body


def test_run_writes_inputs_runs_and_reports(tmp_path: Path) -> None:
    envelope = boot.run(_payload(tmp_path))
    assert envelope["exit_code"] == 0 and envelope["v"] == boot.PROTOCOL
    assert base64.b64decode(envelope["stdout"]).strip() == b"hello"
    assert [f["name"] for f in envelope["files"]] == ["out.txt"]
    assert envelope["duration_ms"] >= 0


def test_run_refuses_unsafe_names(tmp_path: Path) -> None:
    assert boot.run(_payload(tmp_path, files={"../x": _b64(b"")}))["error"] == "unsafe file name"
    assert boot.run(_payload(tmp_path, code_file="/main.py"))["error"] == "unsafe file name"


def test_run_uses_a_given_command(tmp_path: Path) -> None:
    command = [sys.executable, "-I", "-c", "print('given')"]
    envelope = boot.run(_payload(tmp_path, command=command, files={}))
    assert base64.b64decode(envelope["stdout"]).strip() == b"given"


def test_main_reads_stdin_and_prints_one_envelope_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = io.StringIO()
    data = json.dumps(_payload(tmp_path)).encode()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(data)))
    monkeypatch.setattr(sys, "stdout", out)
    assert boot.main() == 0
    envelope = json.loads(out.getvalue().strip().splitlines()[-1])
    assert envelope["exit_code"] == 0


def test_main_refuses_an_unknown_version(monkeypatch: pytest.MonkeyPatch) -> None:
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b'{"v": 99}')))
    monkeypatch.setattr(sys, "stdout", out)
    boot.main()
    assert json.loads(out.getvalue().strip())["error"] == "unknown payload version"


def test_kill_tree_tolerates_a_finished_process(tmp_path: Path) -> None:
    proc = boot._start([*PY, "pass"], str(tmp_path), env(tmp_path))
    proc.wait()
    boot.kill_tree(proc)
    assert proc.returncode == 0
    assert proc.stdout is not None and proc.stderr is not None
    proc.stdout.close()
    proc.stderr.close()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
