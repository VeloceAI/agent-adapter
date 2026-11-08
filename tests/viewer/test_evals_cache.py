"""The Evals folder under stress: locked files, and the bounded cache of parsed reports."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_adapter.viewer import _evals
from agent_adapter.viewer._evals import EvalFolder
from tests.viewer.helpers import request, served
from tests.viewer.test_evals import case, folder, keys, report  # noqa: F401  (fixture)


def lock(monkeypatch: pytest.MonkeyPatch, method: str, name: str) -> None:
    """Make `Path.<method>` raise PermissionError for the file `name`, like an antivirus lock."""
    real = getattr(Path, method)

    def locked(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == name:
            raise PermissionError(name)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, method, locked)


def test_a_locked_report_is_listed_as_unreadable_and_the_rest_still_load(
    folder: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first, second = keys(folder)
    lock(monkeypatch, "read_bytes", f"{first}.json")
    with served(tmp_path / "runs", evals=folder) as server:
        status, body, _ = request(server, "GET", "/api/evals")
    assert status == 200
    rows = {r["key"]: r for r in body["reports"]}
    assert rows[first] == {"key": first, "problem": "cannot be read"}
    assert rows[second]["problem"] == "" and rows[second]["n_cases"] == 2
    assert "cannot read" in caplog.text and "PermissionError" in caplog.text
    assert EvalFolder(folder).report(first) is None


def test_a_locked_file_is_tried_again_on_the_next_list(
    folder: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = keys(folder)[0]
    found = EvalFolder(folder)
    with monkeypatch.context() as patch:
        lock(patch, "read_bytes", f"{first}.json")
        assert found.summaries()[0]["problem"] == "cannot be read"
    assert found.summaries()[0]["problem"] == "", "the lock is gone: it loads"


def test_a_file_that_cannot_be_statted_is_listed_not_fatal(
    folder: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = keys(folder)
    lock(monkeypatch, "stat", f"{first}.json")
    rows = {r["key"]: r for r in EvalFolder(folder).summaries()}
    assert rows[first] == {"key": first, "problem": "cannot be read"}
    assert rows[second]["problem"] == ""
    assert EvalFolder(folder).report(first) is None


def test_the_listing_keeps_summaries_only(folder: Path) -> None:  # noqa: F811
    found = EvalFolder(folder)
    found.summaries()
    assert not found._full, "no full report is kept by a list"
    assert all("cases" not in row for _, row in found._rows.values())


def test_full_reports_are_kept_in_a_small_lru(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "many"
    target.mkdir()
    text = report("shop", case("a", True, 1.0)).to_json()
    for n in range(6):
        (target / f"r{n}.json").write_text(text, encoding="utf-8")
    found = EvalFolder(target)
    for n in range(6):
        assert found.report(f"r{n}") is not None
    assert list(found._full) == ["r2", "r3", "r4", "r5"]
    found.report("r2")  # a hit moves it to the end
    assert list(found._full)[-1] == "r2"
    monkeypatch.setattr(_evals, "FULL_BYTES", (target / "r0.json").stat().st_size * 2)
    found.report("r0")
    assert list(found._full) == ["r2", "r0"], "the byte cap drops the oldest"
    monkeypatch.setattr(_evals, "FULL_BYTES", 1)
    found.report("r1")
    assert list(found._full) == ["r1"], "the newest one stays, whatever its size"


def test_a_changed_file_is_parsed_again(folder: Path) -> None:  # noqa: F811
    key = keys(folder)[0]
    found = EvalFolder(folder)
    assert found.report(key)["name"] == "shop"  # type: ignore[index]
    path = folder / f"{key}.json"
    path.write_text(report("renamed", case("a", True, 1.0)).to_json(), encoding="utf-8")
    os.utime(path, (2_000_000_000, 2_000_000_000))
    assert found.report(key)["name"] == "renamed"  # type: ignore[index]
