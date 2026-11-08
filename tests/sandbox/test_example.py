import runpy
from pathlib import Path

import pytest

import agent_adapter.sandbox as sandbox_pkg
from agent_adapter.sandbox import DockerStatus

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def run(name: str) -> None:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")


def test_code_tool_example_asks_approval_then_answers(capsys: pytest.CaptureFixture[str]) -> None:
    run("sandbox_code_tool.py")
    assert capsys.readouterr().out.splitlines() == [
        "approve? run_code 'prices = [12.5, 8.0, 4.5]\\nprint(sum(prices))'",
        "The basket costs 25.0 in total.",
        "0 5 False",
    ]


def test_hosted_recipes_run_against_stand_ins(capsys: pytest.CaptureFixture[str]) -> None:
    run("sandbox_hosted.py")
    assert capsys.readouterr().out.splitlines() == [
        "e2b 6 0",
        "['create internet=False', 'write data.csv', 'kill']",
        "modal 7",
    ]


def test_docker_example_explains_when_docker_is_missing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = DockerStatus(False, reason="'docker' is not on PATH")
    monkeypatch.setattr(sandbox_pkg, "docker_status", lambda: missing)
    run("sandbox_docker.py")
    first, line = capsys.readouterr().out.splitlines()
    assert first == "docker unavailable: 'docker' is not on PATH"
    assert line.startswith("docker run --rm -i --name aa-sandbox-demo")
    assert "--network none --read-only --cap-drop ALL" in line


def test_concept_page_snippets_come_from_the_examples() -> None:
    page = (ROOT / "docs" / "concepts" / "sandbox.md").read_text(encoding="utf-8")
    parts = {
        "sandbox_code_tool.py": ("sandbox", "tool", "approval", "loop"),
        "sandbox_hosted.py": ("e2b", "modal", "use"),
        "sandbox_docker.py": ("docker",),
    }
    for name, sections in parts.items():
        example = (EXAMPLES / name).read_text(encoding="utf-8")
        for part in sections:
            assert f'--8<-- "examples/{name}:{part}"' in page
            assert f"# --8<-- [start:{part}]" in example


async def test_the_modal_recipe_refuses_input_files() -> None:
    from agent_adapter.core.errors import ToolUsageError
    from agent_adapter.sandbox import CallableSandbox

    recipe = runpy.run_path(str(EXAMPLES / "sandbox_hosted.py"))
    modal = CallableSandbox(recipe["modal_runner"](recipe["fake_modal"](), app=None, image=None))
    with pytest.raises(ToolUsageError, match="no input files"):
        await modal.run("print(1)", files={"a.csv": "x"})
