"""`agent-adapter guide`: prints, writes or slices the packaged guide; `--help` has examples."""

from __future__ import annotations

import io
import runpy
import sys
from pathlib import Path

import pytest

from agent_adapter.cli import main
from agent_adapter.cli.guide import SECTIONS, emit, guide_text, section
from agent_adapter.cli.main import EXAMPLES, command_lines
from agent_adapter.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]


def _run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), out=out)
    return code, out.getvalue()


def test_guide_prints_the_concise_packaged_copy() -> None:
    code, text = _run("guide")
    assert code == 0
    assert text == guide_text() == (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "## API index" not in text


def test_guide_full_adds_the_api_index() -> None:
    code, text = _run("guide", "--full")
    assert code == 0
    assert text == guide_text(full=True) == (ROOT / "docs/llms-full.txt").read_text("utf-8")
    assert "## API index" in text


@pytest.mark.parametrize("name", list(SECTIONS))
def test_guide_section_prints_one_part(name: str) -> None:
    code, text = _run("guide", "--section", name)
    assert code == 0
    assert text.startswith(SECTIONS[name])
    assert text.count("\n## ") == 0  # stops before the next section


def test_guide_out_writes_a_new_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "agent-adapter.md"
    code, text = _run("guide", "--out", str(target))
    assert code == 0 and text == f"wrote {target}\n"
    assert target.read_text(encoding="utf-8") == guide_text()


def test_guide_out_refuses_an_existing_file_without_force(tmp_path: Path, capsys) -> None:
    target = tmp_path / "AGENTS.md"
    target.write_text("my own agent instructions\n", encoding="utf-8")
    assert _run("guide", "--out", str(target))[0] == 2
    assert f"not overwriting {target}; pass --force to replace" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == "my own agent instructions\n"
    assert _run("guide", "--out", str(target), "--force")[0] == 0
    assert target.read_text(encoding="utf-8") == guide_text()


def test_guide_out_with_a_section_writes_only_that_part(tmp_path: Path) -> None:
    target = tmp_path / "install.md"
    assert _run("guide", "--section", "install", "--out", str(target))[0] == 0
    assert target.read_text(encoding="utf-8").startswith("## Install")


def test_guide_out_to_an_unwritable_path_exits_2(tmp_path: Path, capsys) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert _run("guide", "--out", str(blocker / "AGENTS.md"))[0] == 2
    assert "agent-adapter guide:" in capsys.readouterr().err


def test_guide_prints_utf8_on_a_legacy_code_page_stream() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")  # a redirected Windows console
    assert main(["guide"], out=stream) == 0
    assert raw.getvalue().decode("utf-8") == guide_text()
    assert "→" in guide_text()  # the case that used to raise UnicodeEncodeError


def test_emit_writes_text_to_a_text_only_stream() -> None:
    out = io.StringIO()
    emit(out, "→ ok\n")
    assert out.getvalue() == "→ ok\n"


def test_guide_rejects_an_unknown_section() -> None:
    with pytest.raises(SystemExit) as exc:
        _run("guide", "--section", "nope")
    assert exc.value.code == 2
    with pytest.raises(ConfigError, match="unknown section"):
        section(guide_text(), "nope")


def test_guide_section_skips_headings_inside_code_fences() -> None:
    text = "# t\n\n## Install\n\n```bash\n## not a heading\n```\n\n## Import rules\nx\n"
    assert section(text, "install") == "## Install\n\n```bash\n## not a heading\n```\n"


def test_guide_section_missing_from_the_text_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="no '## API index' section"):
        section("# t\n\n## Install\n", "api")


def test_guide_last_section_runs_to_the_end() -> None:
    assert section("## Install\nx\n", "install") == "## Install\nx\n"


def test_help_shows_an_example_for_every_command(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    help_text = capsys.readouterr().out
    assert "examples:" in help_text
    names = [line.split("`")[1].split()[1] for line in command_lines()]
    assert {"init", "doctor", "guide"} <= set(names)
    for name in names:
        assert f"  {EXAMPLES.get(name, f'agent-adapter {name}')}\n" in help_text + "\n", name


def test_python_dash_m_agent_adapter_runs_the_cli(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["agent_adapter", "guide", "--section", "install"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("agent_adapter", run_name="__main__")
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("## Install")
