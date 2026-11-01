"""The Claude Agent SDK gate matches the `claude-agent` extra's floor at patch level."""

import pytest

pytest.importorskip("claude_agent_sdk")

import claude_agent_sdk

from agent_adapter.core.errors import IntegrationError
from agent_adapter.integrations import claude_agent

pytestmark = pytest.mark.contract


@pytest.fixture
def fresh_gate():
    claude_agent._check_frameworks.cache_clear()
    yield
    claude_agent._check_frameworks.cache_clear()


@pytest.mark.usefixtures("fresh_gate")
@pytest.mark.parametrize("version", ["0.2.149", "0.2.0", "0.3.0"])
def test_check_frameworks_refuses_an_sdk_outside_the_extra(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(claude_agent_sdk, "__version__", version)
    with pytest.raises(IntegrationError, match=r"0\.2\.150 to below 0\.3\.0"):
        claude_agent._check_frameworks()


@pytest.mark.usefixtures("fresh_gate")
def test_check_frameworks_accepts_the_extra_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_agent_sdk, "__version__", "0.2.150")
    claude_agent._check_frameworks()
