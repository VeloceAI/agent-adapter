"""Reads a folder of saved eval reports for the Evals view: summaries, one report, a comparison."""

from __future__ import annotations

import logging
import math
import re
import stat
import threading
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent_adapter.evals.report import EvalReport
from agent_adapter.evals.store import DEFAULT_EVALS_FOLDER, REPORT_NAME

logger = logging.getLogger(__name__)

MAX_REPORT_BYTES = 16 * 2**20  # a larger file is listed as unreadable, never parsed
MAX_REPORTS = 500  # the newest files listed
FULL_KEPT = 4  # parsed reports kept for the report and compare routes
FULL_BYTES = 64 * 2**20  # and at most this much of their files
_KEY = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SUMMARY = (
    "name",
    "started_at",
    "wall_s",
    "n_cases",
    "n_passed",
    "n_errors",
    "mean_score",
    "pass_rate",
    "cost_usd",
    "by_tag",
    "by_grader",
)
_UNREADABLE = "cannot be read"

Stamp = tuple[int, int]  # (mtime_ns, size) of a file


def is_key(key: str) -> bool:
    """True for a report key the folder could hold (the file name without `.json`)."""
    return _KEY.match(key) is not None


@dataclass(frozen=True)
class _Parsed:
    report: EvalReport | None
    problem: str  # why `report` is None
    retry: bool = False  # a read error that may pass: never cached


class EvalFolder:
    """A folder of `EvalReport` JSON files; small summaries cached, a few full reports kept.

    Thread-safe. A file is parsed again only when its mtime or size changes; a file that
    cannot be read is listed with a `problem` and tried again on the next list.
    """

    def __init__(self, folder: str | Path = DEFAULT_EVALS_FOLDER) -> None:
        self.folder = Path(folder).resolve()
        self._lock = threading.Lock()
        self._rows: dict[str, tuple[Stamp, dict[str, Any]]] = {}
        self._full: OrderedDict[str, tuple[Stamp, EvalReport]] = OrderedDict()

    def summaries(self) -> list[dict[str, Any]]:
        """The newest reports first: totals per report, or `problem` for an unreadable file.

        Raises:
            OSError: the folder exists but cannot be listed.
        """
        found = [self._row(key, path, stamp) for key, path, stamp in self._files()[:MAX_REPORTS]]
        with self._lock:
            listed = {row["key"] for row in found}
            for gone in set(self._rows) - listed:
                del self._rows[gone]
        return found

    def report(self, key: str) -> dict[str, Any] | None:
        """One report with its totals and cases; None when it is missing or unreadable."""
        found = self._report(key)
        return None if found is None else dict(found.model_dump(mode="json"), key=key)

    def compare(self, base: str, head: str, *, tolerance: float = 0.0) -> dict[str, Any] | None:
        """`head` against the baseline `base` (`EvalReport.compare`); None if either is missing."""
        before, after = self._report(base), self._report(head)
        if before is None or after is None:
            return None
        comparison = after.compare(before, tolerance=tolerance)
        return dict(
            comparison.model_dump(mode="json"),
            regressed=comparison.regressed,
            base=base,
            head=head,
            tolerance=tolerance,
        )

    def _files(self) -> list[tuple[str, Path, Stamp | None]]:
        if not self.folder.is_dir():
            return []
        found: list[tuple[str, Path, Stamp | None]] = []
        for path in self.folder.iterdir():
            match = REPORT_NAME.match(path.name)
            if match is None:
                continue
            try:
                info = path.stat()
            except FileNotFoundError:  # removed while listing
                continue
            except OSError as exc:  # locked (antivirus, sync tools): listed, tried again later
                logger.warning("evals: cannot stat %s (%s)", path.name, type(exc).__name__)
                found.append((match.group(1), path, None))
                continue
            if stat.S_ISREG(info.st_mode):
                found.append((match.group(1), path, (info.st_mtime_ns, info.st_size)))
        found.sort(key=lambda item: (item[2][0] if item[2] else 0, item[0]), reverse=True)
        return found

    def _row(self, key: str, path: Path, stamp: Stamp | None) -> dict[str, Any]:
        """One summary row, from the cache when the file has not changed."""
        if stamp is None:
            return {"key": key, "problem": _UNREADABLE}
        with self._lock:
            cached = self._rows.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        parsed = _parse(path, stamp)
        row: dict[str, Any] = {"key": key, "problem": parsed.problem}
        if parsed.report is not None:
            dumped = parsed.report.model_dump(mode="json", include=set(_SUMMARY))
            row.update({name: dumped[name] for name in _SUMMARY})
        if not parsed.retry:
            with self._lock:
                self._rows[key] = (stamp, row)
        return row

    def _report(self, key: str) -> EvalReport | None:
        """The parsed report `key`, kept in a small LRU and parsed again when its file changes."""
        if not is_key(key):
            return None
        path = self.folder / f"{key}.json"
        try:
            info = path.stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("evals: cannot stat %s (%s)", path.name, type(exc).__name__)
            return None
        if not stat.S_ISREG(info.st_mode):
            return None
        stamp = (info.st_mtime_ns, info.st_size)
        with self._lock:
            kept = self._full.get(key)
            if kept is not None and kept[0] == stamp:
                self._full.move_to_end(key)
                return kept[1]
        parsed = _parse(path, stamp)
        if parsed.report is not None:
            self._keep(key, stamp, parsed.report)
        return parsed.report

    def _keep(self, key: str, stamp: Stamp, report: EvalReport) -> None:
        """Add to the LRU; drop the oldest past `FULL_KEPT` entries or `FULL_BYTES` of files."""
        with self._lock:
            self._full[key] = (stamp, report)
            self._full.move_to_end(key)
            while len(self._full) > FULL_KEPT or (
                len(self._full) > 1 and sum(s[1] for s, _ in self._full.values()) > FULL_BYTES
            ):
                self._full.popitem(last=False)


def _parse(path: Path, stamp: Stamp) -> _Parsed:
    """Parse one report file; a bad, locked or oversized file becomes a `problem`, not an error."""
    if stamp[1] > MAX_REPORT_BYTES:
        logger.warning("evals: %s is over %d MB; not read", path.name, MAX_REPORT_BYTES // 2**20)
        return _Parsed(None, f"over {MAX_REPORT_BYTES // 2**20} MB")
    try:
        report = EvalReport.from_json(path.read_bytes())
    except FileNotFoundError:
        return _Parsed(None, "removed", retry=True)
    except OSError as exc:
        logger.warning("evals: cannot read %s (%s)", path.name, type(exc).__name__)
        return _Parsed(None, _UNREADABLE, retry=True)
    except (ValidationError, ValueError) as exc:
        logger.warning("evals: %s is not an eval report (%s)", path.name, type(exc).__name__)
        return _Parsed(None, "not an eval report")
    if not all(math.isfinite(x) for x in _numbers(report)):
        logger.warning("evals: %s holds a number that is not finite", path.name)
        return _Parsed(None, "not an eval report")
    return _Parsed(report, "")


def _numbers(report: EvalReport) -> list[float]:
    values = [report.wall_s]
    for case in report.cases:
        values += [case.cost_usd, case.latency_s]
        values += [x for x in (case.score, *(g.score for g in case.grades)) if x is not None]
    return values


_COMPARE = re.compile(r"^/api/evals/compare/([A-Za-z0-9_-]{1,128})/([A-Za-z0-9_-]{1,128})$")
_REPORT = re.compile(r"^/api/evals/([A-Za-z0-9_-]{1,128})$")


def evals_route(
    folder: EvalFolder, path: str, query: dict[str, list[str]]
) -> tuple[HTTPStatus, dict[str, Any]]:
    """The status and JSON body of a GET under `/api/evals`.

    Raises:
        OSError: the folder or a report cannot be read.
    """
    if path == "/api/evals":
        return HTTPStatus.OK, {"reports": folder.summaries(), "folder": folder.folder.name}
    if (match := _COMPARE.match(path)) is not None:
        tolerance = _tolerance(query)
        if tolerance is None:
            return HTTPStatus.BAD_REQUEST, {"error": "tolerance must be a number from 0 to 1"}
        found = folder.compare(match.group(1), match.group(2), tolerance=tolerance)
        if found is None:
            return HTTPStatus.NOT_FOUND, {"error": "no such report"}
        return HTTPStatus.OK, found
    if (match := _REPORT.match(path)) is not None:
        report = folder.report(match.group(1))
        if report is None:
            return HTTPStatus.NOT_FOUND, {"error": "no such report"}
        return HTTPStatus.OK, report
    return HTTPStatus.NOT_FOUND, {"error": "not found"}


def _tolerance(query: dict[str, list[str]]) -> float | None:
    """The `tolerance` query value; 0 when absent, None when not a number in [0, 1]."""
    text = query.get("tolerance", ["0"])[0]
    try:
        value = float(text)
    except ValueError:
        return None
    return value if 0 <= value <= 1 else None
