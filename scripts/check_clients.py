"""Run every client in examples/clients against one live server and report which languages pass.

Usage: python scripts/check_clients.py [language ...]   (needs Docker; no arguments runs them all)
"""

import subprocess
import sys
from pathlib import Path

COMPOSE = Path(__file__).resolve().parents[1] / "examples" / "clients" / "compose.yaml"
EXPECTED = ("MCP: Sunny in Paris, 21 C", "REST: Sunny in Paris, 21 C")


def compose(*args: str, project: str = "agent-adapter-clients") -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "compose", "-f", str(COMPOSE), "-p", project, *args]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def languages() -> list[str]:
    services = compose("config", "--services").stdout.split()
    return [name for name in services if name != "server"]


def main() -> int:
    chosen = sys.argv[1:] or languages()
    project = "aa-clients-" + "-".join(chosen) if sys.argv[1:] else "agent-adapter-clients"
    up = compose("up", "-d", "--build", "--wait", "server", project=project)
    if up.returncode:
        print(up.stdout, up.stderr)
        return 1
    failed = []
    try:
        for lang in chosen:
            run = compose("run", "--rm", "--build", lang, project=project)
            lines = run.stdout.splitlines()
            ok = run.returncode == 0 and all(line in lines for line in EXPECTED)
            print(f"{'PASS' if ok else 'FAIL'}  {lang}")
            if not ok:
                failed.append(lang)
                print("\n".join((run.stdout + run.stderr).splitlines()[-25:]))
    finally:
        compose("down", "--remove-orphans", project=project)
    print(f"\n{len(chosen) - len(failed)}/{len(chosen)} languages pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
