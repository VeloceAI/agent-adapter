"""The README hello world and the short approval example run, and the README shows them."""

import runpy
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_adapter.core._default import default_registry
from agent_adapter.exec import defaults

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


@pytest.fixture(autouse=True)
def _isolated_defaults(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The examples register on the default registry; undo that after each test."""
    saved = dict(default_registry._tools)
    monkeypatch.setattr(defaults, "_default", None)
    yield
    default_registry._tools.clear()
    default_registry._tools.update(saved)


def _out(name: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def _body(name: str) -> str:
    """The example without its module docstring, as the README shows it."""
    text = (EXAMPLES / name).read_text(encoding="utf-8")
    return text.split('"""\n', 1)[1].strip()


def test_hello_prints_the_sum(capsys: pytest.CaptureFixture[str]) -> None:
    assert _out("hello.py", capsys) == ["3"]


def test_hello_approval_runs_once_after_resume(capsys: pytest.CaptureFixture[str]) -> None:
    assert _out("hello_approval.py", capsys) == ["refunded 500.0 on o-7"]


def test_readme_hello_world_is_the_tested_example() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"```python\n{_body('hello.py')}\n```" in readme


def test_hello_world_is_short() -> None:
    code = [line for line in _body("hello.py").splitlines() if line.strip()]
    assert len(code) <= 7
