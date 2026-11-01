"""`--export`: every run of a folder (or one run file) in one offline HTML file."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from agent_adapter.core.errors import ConfigError
from agent_adapter.viewer.assets import export_html
from agent_adapter.viewer.runs import RunFolder

logger = logging.getLogger(__name__)

LARGE_BYTES = 50 * 2**20


def export(target: str | Path, out: str | Path, *, run: str | None = None) -> Path:
    """Write the runs under `target` (or only `run`) into the HTML file `out`; return its path.

    Raises:
        ConfigError: `target` is a file that is not a run file, `out` is not an .html path, or
            `run` is not in `target`.
        OSError: the folder cannot be read or `out` cannot be written.
    """
    path = Path(out).resolve()
    if path.suffix.lower() not in (".html", ".htm"):
        raise ConfigError(f"--export needs an .html path, got {out}")
    folder = RunFolder(target)
    folder.refresh()
    runs = [s for s in folder.summaries() if run is None or s["key"] == run]
    if run is not None and not runs:
        raise ConfigError(f"no run {run!r} in {folder.target}")
    details = {s["key"]: folder.detail(s["key"], paged=False) for s in runs}
    data = {"runs": runs, "details": details, "exported_at": time.time()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(export_html(data), encoding="utf-8")
    size = path.stat().st_size
    if size > LARGE_BYTES:
        logger.warning(
            "export %s is %d MB; browsers get slow past %d MB: export one run with --run KEY",
            path.name,
            size // 2**20,
            LARGE_BYTES // 2**20,
        )
    logger.info("exported %d run(s) to %s", len(runs), path)
    return path
