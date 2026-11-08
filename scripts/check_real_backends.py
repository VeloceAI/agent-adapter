"""Run the contract tests that need real services (Docker, Redis, Postgres); report what ran.

Set what you have, then run from the repo root: `python scripts/check_real_backends.py`.
  AA_REDIS_URL  redis://localhost:6379
  AA_PG_URL     postgresql+asyncpg://postgres:x@localhost:5432/postgres
  docker        on PATH, able to run Linux containers
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("check_real_backends")

# (name, how to tell it is available, the pytest selection it unlocks)
CHECKS: list[tuple[str, bool, list[str]]] = [
    ("docker sandbox", shutil.which("docker") is not None, ["-m", "contract", "tests/sandbox"]),
    (
        "redis approval store",
        bool(os.environ.get("AA_REDIS_URL")),
        ["tests/approval/test_store_conformance.py", "-k", "redis-real"],
    ),
    (
        "postgres budget store",
        bool(os.environ.get("AA_PG_URL")),
        ["-m", "contract", "tests/hooks/builtin/budgets/test_hook_review_fixes.py"],
    ),
]


def main() -> int:
    """Run each available check's contract tests; exit 1 if any ran and failed, 2 if none ran."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ran, failed = 0, 0
    for name, available, selection in CHECKS:
        if not available:
            logger.info("skip  %s: not configured (see this script's docstring)", name)
            continue
        args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs"]
        code = subprocess.run([*args, *selection], cwd=ROOT, check=False).returncode
        ran += 1
        failed += code not in (0, 5)  # 5: no tests collected for this marker and path
        logger.info("%s  %s", "ok  " if code in (0, 5) else "FAIL", name)
    if not ran:
        logger.warning("nothing ran: configure at least one backend")
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
