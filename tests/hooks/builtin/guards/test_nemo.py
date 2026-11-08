import enum
import sys
import types

import pytest

from agent_adapter.core.errors import ConfigError, GuardrailBlocked, MissingDependency
from agent_adapter.hooks.builtin.guards import NemoGuard
from tests.hooks.builtin.helpers import llm_inv, llm_response, ok, tool_inv


class RailStatus(enum.StrEnum):
    PASSED = "passed"
    MODIFIED = "modified"
    BLOCKED = "blocked"


class FakeRails:
    """Blocks "jailbreak", rewrites "darn", passes the rest; records the roles it saw."""

    def __init__(self, config=None):
        self.config = config
        self.roles = []

    async def check_async(self, messages):
        message = messages[-1]
        self.roles.append(message["role"])
        content = message["content"]
        if "jailbreak" in content:
            return types.SimpleNamespace(status=RailStatus.BLOCKED, content="", rail="jailbreak")
        if "darn" in content:
            return types.SimpleNamespace(
                status=RailStatus.MODIFIED, content=content.replace("darn", "d*rn"), rail=None
            )
        if "odd" in content:
            return types.SimpleNamespace(status="WEIRD", content="", rail=None)
        if "anon" in content:
            return types.SimpleNamespace(status="BLOCKED", content="")
        return types.SimpleNamespace(status=RailStatus.PASSED, content=content, rail=None)


async def test_inputs_go_as_user_turns_and_outputs_as_assistant_turns():
    rails = FakeRails()
    guard = NemoGuard(rails, on=("args", "result", "llm_request", "llm_response"))
    await guard.before_tool(tool_inv(arguments={"user": "hi"}))
    await guard.after_tool(tool_inv(), ok("fine"))
    await guard.before_llm(llm_inv())
    await guard.after_llm(llm_inv(), llm_response())
    assert rails.roles == [
        "user",
        "user",
        "assistant",
        "user",
        "assistant",
    ]  # a key, then its value


async def test_blocked_names_the_rail():
    guard = NemoGuard(FakeRails(), on="args")
    with pytest.raises(GuardrailBlocked, match="blocked by rail jailbreak"):
        await guard.before_tool(tool_inv(arguments={"user": "try a jailbreak"}))
    with pytest.raises(GuardrailBlocked, match="blocked by rails"):
        await guard.before_tool(tool_inv(arguments={"user": "anon"}))


async def test_modified_redacts_or_blocks():
    redact = NemoGuard(FakeRails(), on="result", action="redact")
    assert (await redact.after_tool(tool_inv(), ok("oh darn"))).value == "oh d*rn"
    with pytest.raises(GuardrailBlocked, match="modified by rails"):
        await NemoGuard(FakeRails(), on="result").after_tool(tool_inv(), ok("oh darn"))


async def test_an_unknown_status_blocks():
    with pytest.raises(GuardrailBlocked, match="guard failed"):
        await NemoGuard(FakeRails(), on="result").after_tool(tool_inv(), ok("odd"))


def test_config_loads_rails(monkeypatch, tmp_path):
    module = types.ModuleType("nemoguardrails")
    module.LLMRails = FakeRails
    module.RailsConfig = types.SimpleNamespace(from_path=lambda path: ("config", path))
    monkeypatch.setitem(sys.modules, "nemoguardrails", module)
    guard = NemoGuard(config=tmp_path)
    assert guard.rails.config == ("config", str(tmp_path))


def test_missing_nemo_names_the_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "nemoguardrails", None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[nemo\]"):
        NemoGuard(config=tmp_path)


@pytest.mark.parametrize("kwargs", [{}, {"rails": FakeRails(), "config": "x"}])
def test_exactly_one_source(kwargs):
    with pytest.raises(ConfigError, match="exactly one"):
        NemoGuard(**kwargs)


@pytest.mark.contract
def test_real_nemo_has_the_check_api():
    nemo = pytest.importorskip("nemoguardrails")
    options = pytest.importorskip("nemoguardrails.rails.llm.options")
    assert callable(getattr(nemo.LLMRails, "check_async", None))
    assert {s.name for s in options.RailStatus} >= {"PASSED", "MODIFIED", "BLOCKED"}


def test_install_hint_works_in_every_shell():
    from agent_adapter.hooks.builtin.guards.nemo import INSTALL

    assert INSTALL == 'pip install "agent-adapter[nemo]"'
