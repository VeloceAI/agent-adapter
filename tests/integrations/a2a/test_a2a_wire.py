"""Pure-Python parts of the A2A bridge: wire format, bounds, host policy, missing SDK."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from agent_adapter.core.errors import MissingDependency
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import build_spec
from agent_adapter.integrations.a2a import INTEGRATION, HostRefused, a2a_app, agent_card
from agent_adapter.integrations.a2a._guard import HostPolicy, is_public
from agent_adapter.integrations.a2a._pending import PendingCall, PendingCalls
from agent_adapter.integrations.a2a._wire import (
    WireError,
    approval_payload,
    check_bounds,
    clip,
    parse_call,
    parse_decision,
    whole_numbers,
)
from agent_adapter.integrations.a2a.agent import owner_ref, scoped_run_id


def test_parse_call_reads_the_first_call_part():
    call = parse_call([{"other": 1}, {"tool": "add", "arguments": {"a": 2.0, "b": [1.0, 1.5]}}])
    assert (call.tool, call.arguments) == ("add", {"a": 2, "b": [1, 1.5]})
    assert parse_call([{"tool": "ping"}]).arguments == {}


@pytest.mark.parametrize(
    ("parts", "error"),
    [
        ([], "send a data part"),
        (["text"], "send a data part"),
        ([{"tool": ""}], "non-empty"),
        ([{"tool": 3}], "non-empty"),
        ([{"tool": "t", "arguments": "x"}], "object"),
        ([{"tool": "t", "arguments": {"x": "y" * 300}}], "limit"),
    ],
)
def test_parse_call_refuses_bad_shapes(parts: list[Any], error: str):
    with pytest.raises(WireError, match=error):
        parse_call(parts, limit=256)


def test_bounds_refuse_deep_and_non_json_payloads():
    deep: Any = 1
    for _ in range(40):
        deep = [deep]
    with pytest.raises(WireError, match="deeper"):
        check_bounds(deep)
    with pytest.raises(WireError, match="plain JSON"):
        check_bounds({"x": float("nan")})
    check_bounds({"x": [], "y": {}})


def test_only_parts_within_the_part_limit_are_read():
    parts = [{"n": i} for i in range(16)] + [{"tool": "late"}]
    with pytest.raises(WireError):
        parse_call(parts)


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        ([{"approval_id": "apr_1", "approved": True, "reason": "ok"}], ("apr_1", True, "ok")),
        ([{"approval_id": "apr_1", "approved": False}], ("apr_1", False, "")),
        ([{"approval_id": "apr_1", "approved": "true"}], None),
        ([{"approval_id": "", "approved": True}], None),
        ([{"approved": True}], None),
        ([], None),
    ],
)
def test_parse_decision_needs_an_explicit_boolean(parts: list[Any], expected: Any):
    decision = parse_decision(parts)
    got = None if decision is None else (decision.approval_id, decision.approved, decision.reason)
    assert got == expected


def test_helpers():
    assert whole_numbers({"a": [2.0, 2.5, float("inf")]}) == {"a": [2, 2.5, float("inf")]}
    assert clip("abc", 2) == "ab ..." and clip("ab", 2) == "ab"
    assert approval_payload("apr_1", "t", client=True)["answer"]["approval_id"] == "apr_1"
    assert isinstance(approval_payload("apr_1", "t", client=False)["answer"], str)
    assert owner_ref("") == "anonymous" and len(owner_ref("a@b.c")) == 12
    assert scoped_run_id("alice", "ctx") != scoped_run_id("bob", "ctx")
    assert scoped_run_id("alice", "ctx") == scoped_run_id("alice", "ctx")


def test_pending_calls_are_taken_once_and_bounded():
    clock = [0.0]
    table = PendingCalls(ttl=10, max_entries=1, clock=lambda: clock[0])
    call = PendingCall("t", {}, "run", "apr_1", "port")
    table.park("alice", "task_1", call)
    assert table.take("bob", "task_1") is None
    assert table.take("alice", "task_1") == call and table.take("alice", "task_1") is None
    table.park("alice", "task_1", call)
    table.park("alice", "task_2", call)  # drops the oldest
    assert len(table) == 1 and table.take("alice", "task_1") is None
    clock[0] = 11.0
    assert table.take("alice", "task_2") is None


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("8.8.8.8", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("169.254.169.254", False),
        ("::1", False),
        ("::ffff:10.0.0.1", False),
        ("fe80::1%eth0", False),
        ("224.0.0.1", False),
        ("2001:4860:4860::8888", True),
        ("64:ff9b::a9fe:a9fe", False),  # NAT64 of 169.254.169.254
        ("64:ff9b::a00:1", False),  # NAT64 of 10.0.0.1
        ("64:ff9b::5db8:d822", True),  # NAT64 of a public address
        ("64:ff9b:1::808:808", False),  # local-use NAT64
        ("::127.0.0.1", False),  # IPv4-compatible
        ("::", False),
        ("fec0::1", False),  # site-local
        ("2002:7f00:1::1", False),  # 6to4 of 127.0.0.1
        ("2002:808:808::1", True),  # 6to4 of 8.8.8.8
        ("2001:0:4136:e378::1", False),  # Teredo
        ("100.64.0.1", False),
    ],
)
def test_is_public(address: str, public: bool):
    assert is_public(address) is public


def _resolves_to(*addresses: str) -> Any:
    async def resolver(host: str, port: int) -> list[str]:
        return list(addresses)

    return resolver


async def test_host_policy():
    public = HostPolicy(resolver=_resolves_to("93.184.216.34"))
    await public.check("https://agent.example/")
    for url in ("http://127.0.0.1/", "http://[::1]:8000/", "file:///etc/passwd", "http:///x"):
        with pytest.raises(HostRefused):
            await public.check(url)
    mixed = HostPolicy(resolver=_resolves_to("93.184.216.34", "10.0.0.1"))
    with pytest.raises(HostRefused, match="private"):
        await mixed.check("https://rebind.example/")
    empty = HostPolicy(resolver=_resolves_to())
    with pytest.raises(HostRefused):
        await empty.check("https://nothing.example/")
    await HostPolicy(trusted_hosts=["Agent.Internal"]).check("http://agent.internal:9/")
    await HostPolicy(allow_private=True).check("http://10.0.0.1/")
    with pytest.raises(HostRefused, match="port"):
        await public.check("http://agent.example:99999/")


async def test_unresolvable_host_is_refused():
    async def resolver(host: str, port: int) -> list[str]:
        raise OSError("no such host")

    with pytest.raises(HostRefused, match="resolve"):
        await HostPolicy(resolver=resolver).check("https://missing.example/")


def test_missing_sdk_names_the_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "a2a", None)
    registry = Registry()
    registry.add(build_spec(lambda: 1, name="one", description="One."))
    for build in (
        lambda: agent_card(registry, url="http://x/"),
        lambda: a2a_app(registry, url="http://x/"),
        lambda: INTEGRATION.export_tool(registry.get("one")),
    ):
        with pytest.raises(MissingDependency, match=r"agent-adapter\[a2a\]"):
            build()
    assert INTEGRATION.name == "a2a" and INTEGRATION.requires == ("a2a",)
