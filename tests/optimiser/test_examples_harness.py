"""The harness examples run offline, print what the how-to page shows, and wrap other frameworks."""

import importlib.util
import runpy
from pathlib import Path

import pytest

from agent_adapter.core.errors import MissingDependency
from agent_adapter.optimiser import NullModel, OptimiserBudget, OptimiserConfig, optimise
from agent_adapter.optimiser.__main__ import dry_run_text

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
PAGE = ROOT / "docs" / "how-to" / "optimise-your-agent.md"
STDOUT = [
    "base 0.25 -> agg1 1.00  (chosen: agg1)",
    "kept    prompt/system",
    "dropped tool/lookup_article/prompt_hint",
    "--- a/prompt/system",
    "You answer customer-support questions. "
    "Always call lookup_article first, then cite the article id.",
]
FRAMEWORKS = [
    ("build_langgraph_host", "langgraph"),
    ("build_openai_agents_host", "agents"),
    ("build_pydantic_ai_host", "pydantic_ai"),
]


@pytest.fixture
def examples(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))


def load(name):
    return runpy.run_path(str(EXAMPLES / f"{name}.py"))


def test_harness_optimise_prints_the_improvement_and_the_applied_text(examples, capsys):
    runpy.run_path(str(EXAMPLES / "harness_optimise.py"), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == STDOUT


async def test_harness_run_keeps_the_budget_identity(examples, tmp_path):
    ns, fakes = load("harness_optimise"), load("harness_fakes")
    config = OptimiserConfig(
        k=1, out_dir=str(tmp_path), lessons_path="", budget=OptimiserBudget(max_case_runs=40)
    )
    result = await optimise(
        ns["build_host"](),
        config,
        proposer=fakes["ScriptedProposer"](),
        model=fakes["FakeFaqModel"](),
    )
    assert result.chosen == "agg1"
    assert (result.base.mean_score, result.probes["agg1"].mean_score) == (0.25, 1.0)
    assert (result.probes["know1"].mean_score, result.probes["proc1"].mean_score) == (1.0, 0.25)
    assert (result.budget.case_runs, result.budget.proposer_calls) == (32, 2)
    lookups = [r.metrics.get("tool_calls.lookup_article", 0.0) for r in result.probes["proc1"].runs]
    assert sum(lookups) == 6.0  # proc1 looked up every unknown topic but cited none


def test_harness_cases_file_has_the_eight_cases(examples):
    cases = load("harness_optimise")["build_host"]().cases()
    assert [c.id for c in cases] == [
        "returns", "refund", "shipping", "tracking", "warranty", "password", "invoice", "cancel",
    ]  # fmt: skip


@pytest.mark.parametrize(("builder", "sdk"), FRAMEWORKS)
def test_framework_hosts_need_their_sdk(examples, builder, sdk):
    if importlib.util.find_spec(sdk) is not None:
        pytest.skip(f"{sdk} is installed")
    with pytest.raises(MissingDependency):
        load("harness_frameworks")[builder](object())


@pytest.mark.parametrize(("builder", "sdk"), FRAMEWORKS)
def test_framework_hosts_build_and_dry_run(examples, builder, sdk):
    pytest.importorskip(sdk)
    host = load("harness_frameworks")[builder](object())
    text = dry_run_text(host, OptimiserConfig(k=1, lessons_path=""))
    assert "Cases (8):" in text and "PROBLEM" not in text
    assert NullModel().provider == "none"


def test_the_page_is_short_shows_the_real_output_and_only_real_snippets():
    text = PAGE.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 150
    assert "\n".join(STDOUT) in text
    for line in text.splitlines():
        if line.startswith('--8<-- "'):
            path, _, section = line.split('"')[1].partition(":")
            source = (ROOT / path).read_text(encoding="utf-8")
            if section:
                assert f"[start:{section}]" in source and f"[end:{section}]" in source, line
