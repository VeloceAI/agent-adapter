"""The runner inside a sandbox: write the code and files, run it with limits, print one JSON line.

Standard library only, and no `agent_adapter` imports: Docker runs this file's source with
the image's own Python (`python3 -c`), and the local sandbox runs it as a script.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import IO, Any

PROTOCOL = 1
CHUNK = 65_536
MAX_WALK = 10_000  # directory entries looked at when collecting produced files
JOIN_S = 2.0  # how long to wait for output from processes the code left behind


def drain(stream: IO[bytes], cap: int, out: dict[str, Any]) -> None:
    """Keep the first `cap` bytes of `stream` and read the rest away, so the writer never blocks."""
    kept = bytearray()
    total = 0
    fd = stream.fileno()
    while True:
        try:
            chunk = os.read(fd, CHUNK)
        except OSError:
            break
        if not chunk:
            break
        total += len(chunk)
        room = cap - len(kept)
        if room > 0:
            kept += chunk[:room]
    with contextlib.suppress(OSError):
        stream.close()
    out["data"] = bytes(kept)
    out["truncated"] = total > len(kept)


def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill `proc` and the processes it started; one already gone is not an error."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass  # the group is already gone; proc.kill() below covers the process itself
    with contextlib.suppress(OSError):
        proc.kill()


def child_env(extra: dict[str, str], workdir: str) -> dict[str, str]:
    """A minimal environment for the code: of the caller's, only PATH and Windows' basics."""
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": workdir,
        "LANG": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if os.name == "nt":
        for key in ("SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "PATHEXT", "COMSPEC"):
            if key in os.environ:
                env[key] = os.environ[key]
    env.update(extra)
    return env


def safe_name(name: object) -> bool:
    """True for a relative name without empty, `.` or `..` parts."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        return False
    return all(part not in ("", ".", "..") for part in name.split("/"))


def write_inputs(workdir: str, code_file: str, code: str, files: dict[str, str]) -> dict[str, str]:
    """Write the code and the base64 input files; return each input's sha256."""
    digests: dict[str, str] = {}
    for name, encoded in files.items():
        data = base64.b64decode(encoded)
        path = os.path.join(workdir, *name.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)
        digests[name] = hashlib.sha256(data).hexdigest()
    with open(os.path.join(workdir, code_file), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(code)
    return digests


def _digest(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            sha.update(block)
    return sha.hexdigest()


def collect(
    workdir: str, skip: str, inputs: dict[str, str], max_files: int, max_bytes: int
) -> tuple[list[dict[str, Any]], bool]:
    """New or changed regular files in `workdir` (no links), bounded; and whether any were cut."""
    found: list[dict[str, Any]] = []
    budget, seen, cut = max_bytes, 0, False
    for root, dirs, names in os.walk(workdir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(names):
            seen += 1
            path = os.path.join(root, name)
            rel = os.path.relpath(path, workdir).replace(os.sep, "/")
            info = os.lstat(path)
            if rel == skip or not stat.S_ISREG(info.st_mode):
                continue
            if rel in inputs and _digest(path) == inputs[rel]:
                continue
            if len(found) >= max_files or seen > MAX_WALK:
                return found, True
            take = min(info.st_size, budget)
            with open(path, "rb") as handle:
                data = handle.read(take)
            budget -= len(data)
            cut = cut or len(data) < info.st_size
            label = rel.encode("utf-8", errors="replace").decode("utf-8")
            encoded = base64.b64encode(data).decode("ascii")
            found.append({"name": label, "size": info.st_size, "data": encoded})
    return found, cut


def _start(command: list[str], workdir: str, env: dict[str, str]) -> subprocess.Popen[bytes]:
    pipes: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE}
    pipes["stderr"] = subprocess.PIPE
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        return subprocess.Popen(command, cwd=workdir, env=env, creationflags=flags, **pipes)
    return subprocess.Popen(command, cwd=workdir, env=env, start_new_session=True, **pipes)


def execute(
    command: list[str], workdir: str, env: dict[str, str], timeout_s: float, cap: int
) -> dict[str, Any]:
    """Run `command` in `workdir`, killed after `timeout_s`, each stream cut at `cap` bytes."""
    started = time.monotonic()
    try:
        proc = _start(command, workdir, env)
    except OSError as exc:
        message = f"cannot start {command[0]}: {type(exc).__name__}\n".encode()
        return {"exit_code": 127, "timed_out": False, "stdout": {}, "stderr": {"data": message}}
    streams: dict[str, dict[str, Any]] = {"stdout": {}, "stderr": {}}
    readers = [
        threading.Thread(target=drain, args=(pipe, cap, streams[key]), daemon=True)
        for key, pipe in (("stdout", proc.stdout), ("stderr", proc.stderr))
    ]
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        exit_code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(proc)
        proc.wait()
        exit_code = -1
    else:
        if sys.platform != "win32":
            kill_tree(proc)  # processes the code left running in the background
    for reader in readers:
        reader.join(JOIN_S)
    result: dict[str, Any] = {"exit_code": exit_code, "timed_out": timed_out, **streams}
    result["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
    return result


def _stream(raw: dict[str, Any]) -> tuple[str, bool]:
    return base64.b64encode(raw.get("data", b"")).decode("ascii"), bool(raw.get("truncated"))


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one payload and return the envelope the host reads."""
    workdir = payload.get("workdir") or os.getcwd()
    files = payload.get("files") or {}
    code_file = payload["code_file"]
    if not all(safe_name(name) for name in (*files, code_file)):
        return {"v": PROTOCOL, "error": "unsafe file name"}
    inputs = write_inputs(workdir, code_file, payload["code"], files)
    command = list(payload.get("command") or [sys.executable, "-I", code_file])
    env = child_env(dict(payload.get("env") or {}), workdir)
    outcome = execute(command, workdir, env, float(payload["timeout_s"]), payload["max_output"])
    produced, cut = collect(
        workdir, code_file, inputs, payload["max_files"], payload["max_file_bytes"]
    )
    stdout, stdout_cut = _stream(outcome["stdout"])
    stderr, stderr_cut = _stream(outcome["stderr"])
    return {
        "v": PROTOCOL,
        "exit_code": outcome["exit_code"],
        "timed_out": outcome["timed_out"],
        "duration_ms": outcome.get("duration_ms", 0.0),
        "stdout": stdout,
        "stdout_truncated": stdout_cut,
        "stderr": stderr,
        "stderr_truncated": stderr_cut,
        "files": produced,
        "files_truncated": cut,
    }


def main() -> int:
    """Read the payload from stdin, run it, print the envelope as one line."""
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("v") != PROTOCOL:
        envelope: dict[str, Any] = {"v": PROTOCOL, "error": "unknown payload version"}
    else:
        envelope = run(payload)
    sys.stdout.write("\n" + json.dumps(envelope) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
