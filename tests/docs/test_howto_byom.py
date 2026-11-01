"""Run the bring-your-own-model how-to example offline and check its page."""

import importlib.util
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "how-to" / "bring-your-own-model.md"
EXAMPLE = ROOT / "examples" / "howto_byom.py"


def test_byom_runs_the_tool_loop_on_the_users_model(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        "stop 2 The rate is 0.92.",
        "[('llm', 'qwen2.5-7b', 180, 0.0), ('tool', 'exchange_rate', 0, 0.0), "
        "('llm', 'qwen2.5-7b', 180, 0.0)]",
        "echo: What is the USD rate? house-13b 0.0000090",
    ]
    has_sdk = importlib.util.find_spec("openai") is not None
    assert lines[3:] == (["llama3.1:8b chat False"] if has_sdk else [])


def test_byom_page_shows_the_real_output_and_snippets() -> None:
    text = PAGE.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 150, "keep a how-to page short"
    source = EXAMPLE.read_text(encoding="utf-8")
    sections = [
        line.split('"')[1].partition(":")[2]
        for line in text.splitlines()
        if line.startswith('--8<-- "examples/howto_byom.py:')
    ]
    assert sections == ["local", "http", "loop", "servers"]
    for section in sections:
        assert f"[start:{section}]" in source and f"[end:{section}]" in source
    assert "llama3.1:8b chat False" in text


def test_byom_example_uses_only_public_import_paths() -> None:
    imports = [
        line
        for line in EXAMPLE.read_text(encoding="utf-8").splitlines()
        if line.startswith(("from agent_adapter", "import agent_adapter"))
    ]
    assert imports
    assert not any("._" in line or "agent_adapter.core" in line for line in imports)


def test_byom_example_prices_the_local_model_as_free_like_the_page_says() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")
    assert '"qwen2.5-7b": {"provider": "local", "free": True}' in source
    assert '"free": true' in PAGE.read_text(encoding="utf-8")
