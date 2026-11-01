"""Run every quality gate in order; stop at the first failure with its exit code."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES: list[tuple[str, list[str]]] = [
    ("tests + coverage", ["-m", "pytest", "--cov", "--cov-report=term-missing"]),
    ("ruff check", ["-m", "ruff", "check", "."]),
    ("ruff format", ["-m", "ruff", "format", "--check", "."]),
    ("mypy", ["-m", "mypy"]),
    ("mkdocs", ["-m", "mkdocs", "build", "--strict"]),
]

logger = logging.getLogger("check")


def main() -> int:
    """Run the gates with this interpreter from the repo root; return the first non-zero code."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for label, args in GATES:
        logger.info("==> %s", label)
        code = subprocess.run([sys.executable, *args], cwd=ROOT, check=False).returncode
        if code != 0:
            logger.error("FAILED: %s (exit %d)", label, code)
            return code
    logger.info("all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
