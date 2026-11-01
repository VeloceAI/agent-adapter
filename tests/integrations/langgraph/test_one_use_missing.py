"""`one_use` without langgraph installed raises MissingDependency naming the extra."""

import sys

import pytest

from agent_adapter.core.errors import MissingDependency
from agent_adapter.integrations.langgraph import one_use


@pytest.mark.parametrize("module", ["langgraph", "langchain_core"])
def test_one_use_without_langgraph_raises_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(MissingDependency, match=r'pip install "agent-adapter\[langgraph\]"'):
        one_use(lambda state: state)
