"""Host executor isolation, anonymous mode, authenticator failures and shared stores."""

import logging
from typing import Any

import pytest

from agent_adapter.approval.builtin import AutoApprove
from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import Forbidden
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.exec.executor import Executor
from agent_adapter.runtime.current import use_context
from agent_adapter.runtime.ids import scoped_id
from agent_adapter.server._approval import ServerApproval, with_metadata
from agent_adapter.server.auth import Principal, RequestInfo, bearer_auth
from agent_adapter.server.service import ToolService
from agent_adapter.tracing.sinks import EventLog
from tests.server.helpers import (
    SECRET,
    TOKENS,
    call,
    invoke,
    make_registry,
    make_service,
)

EMAIL = {"to": "a@example.com", "body": "hi"}


async def test_the_host_executor_is_never_mutated():
    log: list[str] = []
    registry = make_registry(log)
    host_sink, host_port = EventLog(), AutoApprove()
    host = Executor(registry, approval=host_port, sink=host_sink)
    hub = host.sink
    service = make_service(registry, executor=host)
    parked = await invoke(service, "send_email", EMAIL)
    assert parked.status == 202  # the service parks; the host would have auto-approved
    assert host.approval is host_port and host.ledger is None
    # The host's sink is untouched: same hub, same sink, nothing added by the server.
    assert host.sink is hub and hub.base is host_sink and hub.extra == ()
    assert service.executor is not host and service.executor.sink is not hub
    assert isinstance(service.executor.approval, DeferredApproval)
    await host.run("send_email", EMAIL, host.context(run_id="host-run"))
    assert log == ["send a@example.com"]  # still auto-approved on the host
    assert service.events._since("host-run", 0) == []  # host events never reach the stream
    assert [e.type for e in host_sink.events if e.run_id == "host-run"][-1] == "tool.finished"
    assert any(e.type == "tool.pending" for e in host_sink.events)  # host sink still fed


async def test_the_executors_deferred_store_is_reused():
    store = MemoryApprovalStore()
    service = make_service(executor=Executor(approval=DeferredApproval(store)))
    parked = await invoke(service, "send_email", EMAIL)
    assert (await store.get(parked.body["approval_id"])).status == "pending"
    assert service.approvals is store


async def test_anonymous_mode_warns_and_shares_one_identity(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    service = ToolService(make_registry(), allow_anonymous=True)
    assert "accepts anonymous requests" in caplog.text
    first = await invoke(service, "whoami", {}, extra={"run_id": "r1"}, token=None)
    assert first.status == 200 and first.body["result"]["principal"] == ""
    assert first.body["result"]["run_id"] == scoped_id("run_", None, "r1")
    parked = await invoke(service, "send_email", EMAIL, token=None)
    path = f"/approvals/{parked.body['approval_id']}/decide"
    assert (await call(service, "POST", path, body={"approved": True}, token=None)).status == 200


async def test_anonymous_callers_never_join_an_authenticated_principals_run():
    service = make_service(allow_anonymous=True)
    assert (await invoke(service, "add", {"a": 1, "b": 1}, extra={"run_id": "r1"})).status == 200
    anon = await invoke(service, "add", {"a": 1, "b": 1}, extra={"run_id": "r1"}, token=None)
    assert anon.status == 403
    parked = await invoke(service, "send_email", EMAIL)
    seen = await call(service, "GET", f"/approvals/{parked.body['approval_id']}", token=None)
    assert seen.status == 404


@pytest.mark.parametrize(
    ("hook_result", "status"),
    [
        (RuntimeError(SECRET), 500),
        (Forbidden("banned"), 403),
        ("alice", 500),
        (Principal(""), 401),
    ],
)
async def test_authenticator_failures_fail_closed(hook_result: Any, status: int):
    async def hook(request: RequestInfo) -> Any:
        if isinstance(hook_result, BaseException):
            raise hook_result
        return hook_result

    reply = await call(make_service(authenticate=hook), "GET", "/tools")
    assert reply.status == status
    assert SECRET not in str(reply.body)


async def test_two_workers_sharing_a_store_never_merge_runs():
    """Owner tables are per process; scoped run ids keep runs apart on every worker."""
    store = MemoryApprovalStore()
    one = make_service(approvals=store)
    two = make_service(approvals=store)
    parked = await invoke(one, "send_email", EMAIL, extra={"run_id": "r1"})
    bob = await invoke(two, "whoami", {}, extra={"run_id": "r1"}, token="t-bob")
    assert bob.status == 200  # worker two never saw Alice's run ...
    assert bob.body["result"]["run_id"] != scoped_id("run_", "alice@example.com", "r1")
    listed = await call(two, "GET", "/approvals", query={"run_id": "r1"}, token="t-bob")
    assert listed.body["approvals"] == []  # ... and cannot reach her approvals through it
    approval_id = parked.body["approval_id"]
    assert (await call(two, "GET", f"/approvals/{approval_id}", token="t-bob")).status == 404
    decided = await call(two, "POST", f"/approvals/{approval_id}/decide", body={"approved": True})
    assert decided.status == 200  # Alice herself may decide on any worker


async def test_records_without_an_invoker_are_for_approvers_only():
    store = MemoryApprovalStore()
    service = make_service(approvals=store)
    await store.create(
        ApprovalRequest(approval_id="apr_x", run_id="r", call_id="c", tool="send_email")
    )
    assert (await call(service, "GET", "/approvals/apr_x")).status == 404
    assert (await call(service, "GET", "/approvals/apr_x", token="t-reviewer")).status == 200


def test_metadata_is_recorded_only_inside_the_calls_own_context():
    request = ApprovalRequest(approval_id="a", run_id="r", call_id="c", tool="t")
    assert with_metadata(request) is request
    ctx = ToolContext(run_id="r", call_id="other", extras={"principal": Principal("p")})
    with use_context(ctx):
        assert with_metadata(request).metadata == {}
    mcp_ctx = ToolContext(run_id="r", call_id="c", thread_id="th", extras={"principal": "p"})
    with use_context(mcp_ctx):
        assert with_metadata(request).metadata == {
            "principal": "p",
            "principal_scopes": [],
            "thread_id": "th",
        }
    anon_ctx = ToolContext(run_id="r", call_id="c", extras={"principal": Principal("")})
    with use_context(anon_ctx):
        assert with_metadata(request).metadata == {"anonymous": True}


async def test_server_approval_records_metadata_but_binds_the_same_digest():
    store = MemoryApprovalStore()
    port = ServerApproval(store)
    request = ApprovalRequest(approval_id="a", run_id="r", call_id="c", tool="t")
    ctx = ToolContext(run_id="r", call_id="c", extras={"principal": Principal("p")})
    with use_context(ctx):
        record = await port.prepare(request)
    assert record.request.metadata == {"principal": "p", "principal_scopes": []}
    again = await port.prepare(request)  # outside the context: same record, same digest
    assert again.approval_id == record.approval_id


def test_bearer_tokens_resolve_principals():
    assert TOKENS["t-alice"].id == "alice@example.com"
    assert callable(bearer_auth(TOKENS))
