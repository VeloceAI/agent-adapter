"""Without mcp installed, the server builders raise MissingDependency before logging anything."""

import logging
import sys

import pytest

from agent_adapter.core.errors import MissingDependency
from agent_adapter.core.registry import Registry
from agent_adapter.integrations.mcp import mcp_server, streamable_http_app


@pytest.fixture
def no_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mcp", None)


@pytest.mark.usefixtures("no_mcp")
@pytest.mark.parametrize("allow_anonymous", [False, True])
def test_streamable_http_app_without_mcp_raises_before_auth_warnings(
    caplog: pytest.LogCaptureFixture, allow_anonymous: bool
) -> None:
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(MissingDependency, match=r"agent-adapter\[mcp\]"),
    ):
        streamable_http_app(Registry(), allow_anonymous=allow_anonymous)
    assert caplog.records == []


@pytest.mark.usefixtures("no_mcp")
def test_mcp_server_without_mcp_raises_before_option_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(MissingDependency, match=r"agent-adapter\[mcp\]"),
    ):
        mcp_server(Registry(), name="tax", client_approval=True, request_state_keys=[b"k" * 32])
    assert caplog.records == []
