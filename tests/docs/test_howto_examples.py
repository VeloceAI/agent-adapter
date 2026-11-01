"""Run every how-to example offline and check what it prints."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
HOW_TO = ROOT / "docs" / "how-to"
PAGES = (
    "attach-anywhere.md",
    "wrap-legacy-code.md",
    "parallel-retrieval.md",
    "human-in-the-loop.md",
    "budgets.md",
    "caching.md",
    "sync-backends.md",
    "testing-tools.md",
)


def _lines(name: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def test_attach_runs_one_tool_on_every_surface(capsys: pytest.CaptureFixture[str]) -> None:
    lines = _lines("howto_attach.py", capsys)
    assert lines[0] == "Article HC-101: how to fix it"
    assert lines[1] == "['Article HC-102: how to fix it', 'Article HC-201: how to fix it']"
    assert lines[2] == "denied: no reviewer in this script"
    assert json.loads(lines[3]) == ["article"]
    assert json.loads(lines[4]) == ["description", "input_schema", "name"]
    assert lines[5] == "Article HC-301: how to fix it"


def test_wrap_legacy_exposes_methods_and_hides_raw_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    lines = _lines("howto_wrap_legacy.py", capsys)
    assert lines[0] == "['invoice_find', 'invoice_total_open', 'invoice_void', 'invoice_lookup']"
    assert lines[1] == "{'amount': 1200.0, 'status': 'open'}"
    assert lines[2] == "error tool-failed tool 'invoice_find' failed after 1 attempt"
    assert "KeyError" not in lines[2]
    assert lines[3] == "tool-error no invoice X; ask the user for the number"
    assert lines[4:] == ["INV-1 voided: duplicate", "300.0"]


def test_parallel_fans_out_merges_and_stops_early(capsys: pytest.CaptureFixture[str]) -> None:
    lines = _lines("howto_parallel.py", capsys)
    assert lines == [
        "['ok', 'ok', 'error']",
        "['returns policy', 'returns label', 'returns similar']",
        "keyword_search: returns policy",
        "['returns label', 'returns policy']",
        "[14, 13]",
    ]


def test_hitl_pauses_edits_denies_and_uses_each_decision_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    lines = _lines("howto_hitl.py", capsys)
    assert lines == [
        "refunded 20.0 on o-1",
        "pending approval-pending",
        "refund {'order_id': 'o-7', 'amount': 500.0}",
        "first decision wins",
        "ok refunded 450.0 on o-7",
        "True",
        "pending True",
        "error approval denied for tool 'refund': refund policy: max 500",
        "[20.0, 450.0]",
    ]


def test_budgets_stop_warn_and_cap_calls(capsys: pytest.CaptureFixture[str]) -> None:
    lines = _lines("howto_budgets.py", capsys)
    assert lines == [
        "3 run 0.5 0.6",
        "0.6 4 1",
        "0.5 1",
        "sent",
        "run-limit-exceeded",
        "max_calls reached after 9 geocodes",
    ]


def test_caching_hits_tools_and_llm_responses(capsys: pytest.CaptureFixture[str]) -> None:
    lines = _lines("howto_caching.py", capsys)
    assert lines == ["False True 0 1", "2", "0.4", "False True False 2", "0.33"]


def test_sync_backends_run_on_fresh_loops(capsys: pytest.CaptureFixture[str]) -> None:
    lines = _lines("howto_sync.py", capsys)
    assert lines == [
        "{'item': 'laptop', 'rate': 18.0}",
        "[18.0, 5.0, 18.0]",
        "{'item': 'book', 'rate': 5.0}",
        "3",
        "True True",
    ]


def test_testing_recipes_pass(capsys: pytest.CaptureFixture[str]) -> None:
    lines = _lines("howto_testing.py", capsys)
    assert lines == [
        "test_schema_and_validation ok",
        "test_retries_without_waiting ok",
        "test_approval_paths ok",
        "test_agent_loop_with_a_fake_model ok",
    ]


@pytest.mark.parametrize("page", PAGES)
def test_how_to_pages_include_only_real_snippets(page: str) -> None:
    text = (HOW_TO / page).read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 160, "keep a how-to page short"
    for line in text.splitlines():
        if line.startswith('--8<-- "'):
            target = line.split('"')[1]
            path, _, section = target.partition(":")
            source = (ROOT / path).read_text(encoding="utf-8")
            if section:
                assert f"[start:{section}]" in source, target
