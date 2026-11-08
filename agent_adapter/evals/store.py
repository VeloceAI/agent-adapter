"""Save eval reports as JSON files in a folder the viewer's Evals view lists."""

from __future__ import annotations

import logging
import os
import re
import secrets
from pathlib import Path

from agent_adapter.evals.report import EvalReport

logger = logging.getLogger(__name__)

DEFAULT_EVALS_FOLDER = ".agent_adapter/evals"
# <stamp>-<name>-<random>.json; the viewer lists only names of this shape.
REPORT_NAME = re.compile(r"^([A-Za-z0-9_-]{1,128})\.json$")
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")
_NAME_CHARS = 40


def report_key(report: EvalReport) -> str:
    """A new file key for `report`: start time, a safe form of its name, and a random part."""
    stamp = re.sub(r"[^0-9T]", "", report.started_at)[:15] or "report"
    slug = _UNSAFE.sub("-", report.name).strip("-")[:_NAME_CHARS].strip("-") or "eval"
    return f"{stamp}-{slug}-{secrets.token_hex(3)}"


def save_report(report: EvalReport, folder: str | os.PathLike[str] = DEFAULT_EVALS_FOLDER) -> Path:
    """Write `report` as `<folder>/<key>.json` (atomically) and return the path.

    Raises:
        OSError: the folder cannot be created or the file cannot be written.
    """
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{report_key(report)}.json"
    partial = path.with_name(f".{path.stem}.partial")
    try:
        partial.write_text(report.to_json(), encoding="utf-8")
        os.replace(partial, path)
    except OSError:
        partial.unlink(missing_ok=True)
        raise
    logger.info("eval %s: report saved to %s", report.name, path)
    return path
