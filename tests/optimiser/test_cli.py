import runpy
import sys

import pytest

from agent_adapter.core.errors import ConfigError, MissingDependency
from agent_adapter.models.gemini import GeminiModel
from agent_adapter.optimiser import __main__ as cli
from tests.optimiser.fakes import FakeHost, ScriptedProposer, ToyAgent, keep_base, proposal

HOST = "tests.optimiser.fakes:FakeHost"


def build_bad_host():
    return object()


def test_dry_run_prints_surface_cases_and_identity_without_model_calls(capsys, monkeypatch):
    def forbidden(spec):
        raise AssertionError("dry run built a model")

    monkeypatch.setattr(cli, "model_from_spec", forbidden)
    code = cli.main(["--host", HOST, "--k", "2", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Surface (3 paths, domain tax):" in out
    assert "  prompt/system  25 chars (cap 12000)" in out
    assert "Cases (3):\n  q1\n  q2\n  q3" in out
    assert "case runs       (1 + 3k) x N = 21" in out
    assert "proposer calls  2k = 4" in out
    assert "merge calls     0 (greedy compose)" in out
    assert "No model was called" not in out and "no model was called" in out


def test_dry_run_with_llm_compose_and_no_lessons(capsys):
    code = cli.main(["--host", HOST, "--compose", "llm", "--dry-run", "--lessons", ""])
    out = capsys.readouterr().out
    assert code == 0 and "at most 1 (llm)" in out and "reflection      0" in out


@pytest.fixture
def fake_models(monkeypatch):
    agent = ToyAgent()
    scripts = {
        "keep": ScriptedProposer({"proc1": keep_base(), "know1": keep_base()}),
        "cite": ScriptedProposer(
            {"proc1": keep_base(), "know1": proposal(("prompt/system", "cite the section"))}
        ),
    }

    def build(spec):
        return agent.port if spec == "fake:agent" else scripts[spec.split(":")[1]].port

    monkeypatch.setattr(cli, "model_from_spec", build)
    return scripts


def test_run_exits_0_with_a_patch_and_2_for_keep_base(tmp_path, capsys, fake_models):
    common = ["--host", HOST, "--model", "fake:agent", "--lessons", ""]
    code = cli.main([*common, "--proposer-model", "fake:cite", "--out", str(tmp_path / "a")])
    assert code == 0
    assert "chosen: agg1; case runs 12; proposer calls 2" in capsys.readouterr().out
    assert (tmp_path / "a" / "harness.patch").read_text("utf-8")
    code = cli.main([*common, "--proposer-model", "fake:keep", "--out", str(tmp_path / "b")])
    assert code == 2
    assert "chosen: KEEP_BASE" in capsys.readouterr().out


def test_budget_flags_reach_the_config(tmp_path, capsys, fake_models):
    code = cli.main(
        [
            "--host",
            HOST,
            "--model",
            "fake:agent",
            "--proposer-model",
            "fake:cite",
            "--out",
            str(tmp_path),
            "--max-case-runs",
            "3",
            "--budget-usd",
            "9",
            "--lessons",
            "",
        ]
    )
    assert code == 3  # stopped before a final was chosen: not the same as KEEP_BASE
    assert "stopped: max_case_runs" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--host", "nocolon", "--dry-run"], "--host must be module:callable"),
        (["--host", "tests.optimiser.nope:x", "--dry-run"], "cannot load host"),
        (["--host", "tests.optimiser.fakes:missing", "--dry-run"], "cannot load host"),
        (["--host", "tests.optimiser.test_cli:build_bad_host", "--dry-run"], "not a Host"),
        (["--host", HOST], "--model is required"),
        (["--host", HOST, "--k", "x"], "usage error"),
        (["--host", HOST, "--model", "nope"], "provider:model"),
        (["--host", HOST, "--model", "mystery:m"], "unknown model provider"),
        (["--host", HOST, "--model", "openai:gpt-5", "--k", "0", "--lessons", ""], "error:"),
    ],
)
def test_errors_exit_1(argv, message, capsys):
    assert cli.main(argv) == 1
    assert message in capsys.readouterr().err


def test_unexpected_errors_exit_1(capsys, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(cli, "load_host", boom)
    assert cli.main(["--host", HOST, "--dry-run"]) == 1
    assert "RuntimeError: bug" in capsys.readouterr().err


def test_model_from_spec_builds_provider_ports():
    from agent_adapter.models.anthropic import AnthropicModel
    from agent_adapter.models.openai import OpenAIModel

    built = 0
    for spec, cls in (
        ("gemini:gemini-2.5-pro", GeminiModel),
        ("openai:gpt-5", OpenAIModel),
        ("anthropic:claude-x", AnthropicModel),
    ):
        try:
            port = cli.model_from_spec(spec)
        except MissingDependency:
            continue  # that provider's SDK extra is not installed here
        assert isinstance(port, cls) and port.model == spec.split(":")[1]
        built += 1
    assert built <= 3
    with pytest.raises(ConfigError):
        cli.model_from_spec("gpt-5")


def test_module_entry_point_runs_main(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["optimiser", "--host", HOST, "--dry-run"])
    monkeypatch.delitem(sys.modules, "agent_adapter.optimiser.__main__", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("agent_adapter.optimiser", run_name="__main__")
    assert exit_info.value.code == 0
    assert "Budget identity" in capsys.readouterr().out


def test_load_host_accepts_an_instance_attribute(monkeypatch):
    import tests.optimiser.fakes as fakes

    monkeypatch.setattr(fakes, "HOST_INSTANCE", FakeHost(), raising=False)
    assert isinstance(cli.load_host("tests.optimiser.fakes:HOST_INSTANCE"), FakeHost)


def build_odd_host():
    return FakeHost(files={"prompt/system": "s", "bad path": "x"})


def test_dry_run_lists_off_grammar_paths(capsys):
    assert cli.main(["--host", "tests.optimiser.test_cli:build_odd_host", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "  bad path  1 chars" in out
    assert "  PROBLEM: bad path: not an editable path" in out


def test_dry_run_checks_the_config(capsys):
    assert cli.main(["--host", HOST, "--k", "0", "--dry-run"]) == 1
    assert "k must be >= 1" in capsys.readouterr().err


def test_a_bad_api_key_exits_1(tmp_path, capsys, monkeypatch):
    from agent_adapter.core.errors import ModelAuthError
    from agent_adapter.models.callable import CallableModel

    async def denied(request):
        raise ModelAuthError("invalid key", provider="openai")

    monkeypatch.setattr(cli, "model_from_spec", lambda spec: CallableModel(denied))
    code = cli.main(["--host", HOST, "--model", "x:y", "--out", str(tmp_path), "--lessons", ""])
    assert code == 1
    assert "model-auth-failed" in capsys.readouterr().err


# --- --model none ---------------------------------------------------------------------------


async def _own_model_agent(tree, case, model):
    """An agent that brings its own model: it never touches the one it is given."""
    return f"{case.input}: see section 80C" if "cite" in tree.files["prompt/system"] else "no idea"


def build_own_model_host():
    from agent_adapter.optimiser import CallableHost
    from agent_adapter.optimiser.graders import contains
    from tests.optimiser.fakes import CASES

    return CallableHost(
        _own_model_agent, surface={"prompt/system": "Answer."}, cases=CASES, judge=contains()
    )


@pytest.fixture
def none_models(monkeypatch):
    real = cli.model_from_spec
    proposer = ScriptedProposer({"know1": proposal(("prompt/system", "cite the section"))})

    def build(spec):
        return real(spec) if spec == "none" else proposer.port

    monkeypatch.setattr(cli, "model_from_spec", build)


def test_model_none_is_a_null_model():
    from agent_adapter.optimiser import NullModel

    assert isinstance(cli.model_from_spec("none"), NullModel)


def test_model_none_dry_run_exits_0(capsys):
    assert cli.main(["--host", HOST, "--model", "none", "--dry-run"]) == 0
    assert "no model was called" in capsys.readouterr().out


def test_model_none_needs_a_proposer_model(capsys):
    assert cli.main(["--host", HOST, "--model", "none"]) == 1
    assert "--model none needs --proposer-model" in capsys.readouterr().err


def test_model_none_drives_an_agent_with_its_own_model(tmp_path, capsys, none_models):
    host = "tests.optimiser.test_cli:build_own_model_host"
    argv = ["--host", host, "--model", "none", "--proposer-model", "fake:cite"]
    code = cli.main([*argv, "--out", str(tmp_path), "--lessons", ""])
    assert code == 0
    assert "chosen: agg1" in capsys.readouterr().out


def test_model_none_with_a_model_using_host_exits_1(tmp_path, capsys, none_models):
    argv = ["--host", HOST, "--model", "none", "--proposer-model", "fake:cite"]
    code = cli.main([*argv, "--out", str(tmp_path), "--lessons", ""])
    assert code == 1
    err = capsys.readouterr().err
    assert "config-invalid" in err and "without --model" in err
