"""Run every cookbook example offline and check what it prints against its page."""

import json
import re
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
COOKBOOK = ROOT / "docs" / "cookbook"
SNIPPET = re.compile(r'--8<-- "(examples/[\w/]+\.py)')


def _lines(name: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def _page_output(page: str) -> list[str]:
    """Every line of the page's ```text blocks, in order."""
    text = (COOKBOOK / page).read_text(encoding="utf-8")
    blocks = re.findall(r"```text\n(.*?)```", text, re.S)
    return [line for block in blocks for line in block.splitlines()]


@pytest.mark.parametrize(
    ("example", "page"),
    [
        ("cookbook_retry.py", "retry-flaky-tool.md"),
        ("cookbook_user_budget.py", "spend-per-user.md"),
        ("cookbook_parallel_timeout.py", "parallel-deadline.md"),
        ("cookbook_approval_notify.py", "approval-notify.md"),
        ("cookbook_llm_cache.py", "cache-llm.md"),
        ("cookbook_fakes.py", "test-with-fakes.md"),
        ("cookbook_tenants.py", "tenant-registry.md"),
    ],
)
def test_recipe_prints_what_its_page_shows(
    example: str, page: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _lines(example, capsys) == _page_output(page)


def test_redact_masks_logs_results_and_your_own_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    lines = _lines("cookbook_redact.py", capsys)
    page = _page_output("redact.md")
    assert lines[0] == page[0]
    assert lines[2] == page[2]
    event, body = lines[1].split(" ", 1)
    fields = json.loads(body)
    assert event == "tool.start"
    assert fields["args"] == {"email": "***", "api_key": "***"}
    assert fields["run_id"].startswith("run_") and fields["call_id"].startswith("call_")
    assert "sk-123" not in lines[1] and "a@example.com" not in lines[1]


def test_retry_recipe_never_retries_a_plain_tool_error() -> None:
    ns = runpy.run_path(str(EXAMPLES / "cookbook_retry.py"))
    assert ns["UpstreamBusy"].retryable is True
    assert ns["ToolError"].retryable is False


async def test_approval_callback_refuses_an_unknown_reviewer() -> None:
    ns = runpy.run_path(str(EXAMPLES / "cookbook_approval_notify.py"))
    body = json.dumps({"approval_id": "apr_x", "user": "U_EVE", "value": "approve"}).encode()
    assert await ns["on_button"](body, ns["sign"](body)) == "403 not a reviewer"


def test_index_lists_every_recipe_page() -> None:
    index = (COOKBOOK / "index.md").read_text(encoding="utf-8")
    pages = {p.name for p in COOKBOOK.glob("*.md")} - {"index.md"}
    assert pages and all(f"]({page})" in index for page in pages)


def test_every_recipe_page_includes_only_existing_cookbook_examples() -> None:
    for page in COOKBOOK.glob("*.md"):
        for example in SNIPPET.findall(page.read_text(encoding="utf-8")):
            assert example.startswith("examples/cookbook_"), (page.name, example)
            assert (ROOT / example).is_file(), (page.name, example)
