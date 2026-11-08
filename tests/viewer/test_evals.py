"""The Evals view's server side: the report folder, its routes, and exports without it."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.evals import CaseResult, EvalReport, GradeResult, save_report
from agent_adapter.viewer import assets
from agent_adapter.viewer._evals import MAX_REPORTS, EvalFolder, is_key
from tests.viewer.helpers import request, served


def case(case_id: str, passed: bool, score: float, **fields: Any) -> CaseResult:
    grade = GradeResult(grader="answer", score=score, passed=passed, notes="why")
    return CaseResult(case_id=case_id, passed=passed, score=score, grades=(grade,), **fields)


def report(name: str, *cases: CaseResult) -> EvalReport:
    return EvalReport(name=name, started_at="2026-11-05T10:00:00+00:00", wall_s=1.5, cases=cases)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    base = report("shop", case("a", True, 1.0, tags=("orders",)), case("b", True, 1.0))
    head = report("shop", case("a", False, 0.0, run_id="run_1", input="hi"), case("c", True, 1.0))
    target = tmp_path / "evals"
    first = save_report(base, target)
    os.utime(first, (1_000, 1_000))  # older, so listed second
    save_report(head, target)
    return target


def keys(target: Path) -> list[str]:
    return [row["key"] for row in EvalFolder(target).summaries()]


def test_summaries_list_the_newest_first_with_totals(folder: Path) -> None:
    rows = EvalFolder(folder).summaries()
    assert [r["n_passed"] for r in rows] == [1, 2]
    newest = rows[0]
    assert newest["problem"] == "" and newest["name"] == "shop" and newest["n_cases"] == 2
    assert newest["mean_score"] == 0.5 and newest["pass_rate"] == 0.5
    assert rows[1]["by_tag"] == {"orders": {"n": 1, "n_passed": 1, "mean_score": 1.0}}
    assert "cases" not in newest, "a summary stays small"


def test_a_report_has_its_cases_and_a_missing_one_is_none(folder: Path) -> None:
    found = EvalFolder(folder)
    key = keys(folder)[0]
    detail = found.report(key)
    assert detail is not None and detail["key"] == key
    assert detail["cases"][0]["run_id"] == "run_1" and detail["cases"][0]["input"] == "hi"
    for bad in ("nope", "../evals", "", "a/b", "x" * 129):
        assert found.report(bad) is None
    (folder / "dir.json").mkdir()
    assert found.report("dir") is None


def test_compare_uses_the_report_comparison(folder: Path) -> None:
    head, base = keys(folder)
    found = EvalFolder(folder).compare(base, head)
    assert found is not None and found["regressed"] is True
    assert [c["case_id"] for c in found["regressions"]] == ["a"]
    assert found["added"] == ["c"] and found["removed"] == ["b"]
    assert found["score_delta"] == pytest.approx(-0.5)
    assert EvalFolder(folder).compare(base, "missing") is None


@pytest.mark.parametrize(
    ("text", "problem"),
    [("{not json", "not an eval report"), ('{"name": 1}', "not an eval report")],
)
def test_a_bad_file_is_listed_with_its_problem_never_parsed_as_a_report(
    folder: Path, text: str, problem: str
) -> None:
    (folder / "zzz-bad.json").write_text(text, encoding="utf-8")
    (folder / "notes.txt").write_text("ignored", encoding="utf-8")
    (folder / ".hidden.partial").write_text("ignored", encoding="utf-8")
    rows = {r["key"]: r for r in EvalFolder(folder).summaries()}
    assert rows["zzz-bad"] == {"key": "zzz-bad", "problem": problem}
    assert len(rows) == 3
    assert EvalFolder(folder).report("zzz-bad") is None


def test_an_oversized_or_non_finite_report_is_refused(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = report("shop", case("a", True, 1.0)).to_json().replace('"wall_s": 1.5', '"wall_s": NaN')
    (folder / "nan.json").write_text(text, encoding="utf-8")
    rows = {r["key"]: r for r in EvalFolder(folder).summaries()}
    assert rows["nan"]["problem"] == "not an eval report"
    monkeypatch.setattr("agent_adapter.viewer._evals.MAX_REPORT_BYTES", 10)
    rows = {r["key"]: r for r in EvalFolder(folder).summaries()}
    assert all(r["problem"] == "over 0 MB" for r in rows.values())


def test_files_are_read_again_only_when_they_change(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[str] = []
    real = Path.read_bytes

    def counted(self: Path) -> bytes:
        reads.append(self.name)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", counted)
    found = EvalFolder(folder)
    found.summaries()
    found.summaries()
    assert len(reads) == 2
    key = keys(folder)[0]
    path = folder / f"{key}.json"
    path.write_text(report("renamed", case("a", True, 1.0)).to_json(), encoding="utf-8")
    assert found.summaries()[0]["name"] == "renamed"
    path.unlink()
    assert [r["key"] for r in found.summaries()] == keys(folder)


def test_a_missing_folder_lists_nothing_and_the_list_is_capped(tmp_path: Path) -> None:
    assert EvalFolder(tmp_path / "none").summaries() == []
    target = tmp_path / "many"
    target.mkdir()
    for n in range(MAX_REPORTS + 2):
        (target / f"r{n}.json").write_text("{}", encoding="utf-8")
    assert len(EvalFolder(target).summaries()) == MAX_REPORTS


def test_keys_are_plain_names() -> None:
    assert is_key("20261105T100000-shop-a1b2c3")
    for bad in ("", "a.b", "../x", "a b", "x" * 129):
        assert not is_key(bad)


def test_the_routes_need_the_token_and_answer_json(folder: Path, tmp_path: Path) -> None:
    head, base = keys(folder)
    with served(tmp_path / "runs", evals=folder) as server:
        assert request(server, "GET", "/api/evals", auth=False)[0] == 401
        status, body, headers = request(server, "GET", "/api/evals")
        assert status == 200 and [r["key"] for r in body["reports"]] == [head, base]
        assert (
            body["folder"] == "evals" and "default-src 'none'" in headers["content-security-policy"]
        )
        status, body, _ = request(server, "GET", f"/api/evals/{head}")
        assert status == 200 and body["name"] == "shop"
        status, body, _ = request(server, "GET", f"/api/evals/compare/{base}/{head}?tolerance=0.1")
        assert status == 200 and body["regressed"] and body["tolerance"] == 0.1
        for path in (
            f"/api/evals/compare/{base}/{head}?tolerance=2",
            f"/api/evals/compare/{base}/{head}?tolerance=x",
        ):
            assert request(server, "GET", path)[0] == 400
        for path in (
            "/api/evals/nope",
            f"/api/evals/compare/{base}/nope",
            "/api/evals/a/b/c",
            "/api/evals/..%2Fx",
        ):
            assert request(server, "GET", path)[0] == 404
        assert request(server, "POST", "/api/evals", body=b"{}")[0] == 404


def test_an_unreadable_folder_is_a_500_not_a_crash(
    folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: EvalFolder) -> list[object]:
        raise PermissionError("denied")

    monkeypatch.setattr(EvalFolder, "summaries", boom)
    with served(tmp_path / "runs", evals=folder) as server:
        status, body, _ = request(server, "GET", "/api/evals")
    assert status == 500 and body == {"error": "the viewer could not read it"}


def test_the_evals_files_are_served_live_and_left_out_of_every_export() -> None:
    for name in (*assets.EVAL_SCRIPTS, *assets.EVAL_STYLES):
        assert assets.static_file(name) is not None
        assert f"static/{name}" in assets.page("x", approvals=False)
    data = {"runs": [], "details": {}, "chat": {"info": {}, "chats": []}}
    for html in (assets.export_html(data), assets.export_chat_demo_html(data)):
        assert "evals" not in html.replace("eval_cases", ""), "no Evals view offline"


def test_the_evals_scripts_never_parse_html() -> None:
    for name in assets.EVAL_SCRIPTS:
        text = (assets.STATIC / name).read_text(encoding="utf-8")
        assert "innerHTML" not in text and "insertAdjacentHTML" not in text
        assert not re.search(r"(?<![\w.])eval\(", text) and "new Function" not in text


def test_the_chat_server_lists_the_same_evals_folder(folder: Path, tmp_path: Path) -> None:
    from agent_adapter.viewer.serving import make_chat_server

    chat = make_chat_server(
        agent=lambda message, session: "hi",
        folder=tmp_path / "runs",
        port=0,
        chat_dir=tmp_path / "chat",
        evals=folder,
    )
    try:
        assert chat.evals.folder == folder.resolve()
    finally:
        chat.server_close()


def test_a_file_removed_while_listing_or_reading_is_skipped(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gone(self: Path) -> bytes:
        raise FileNotFoundError(self.name)

    monkeypatch.setattr(Path, "read_bytes", gone)
    assert {r["problem"] for r in EvalFolder(folder).summaries()} == {"removed"}
    real_stat = Path.stat

    def stat(self: Path, **kwargs: Any) -> os.stat_result:
        if self.suffix == ".json":
            raise FileNotFoundError(self.name)
        return real_stat(self, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    assert EvalFolder(folder).summaries() == []
