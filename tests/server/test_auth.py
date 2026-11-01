from typing import Any

import pytest

from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.server.auth import (
    APPROVER_SCOPE,
    Principal,
    RequestInfo,
    allow_authenticated,
    bearer_auth,
    bearer_token,
    owner_or_approver,
    principal_ref,
)

ALICE = Principal("alice")


def request(authorization: str | None = None, **headers: str) -> RequestInfo:
    if authorization is not None:
        headers["Authorization"] = authorization
    return RequestInfo("GET", "/tools", headers, {}, None)


def record(**metadata: Any) -> ApprovalRecord:
    req = ApprovalRequest(approval_id="apr_1", run_id="r", call_id="c", tool="t", metadata=metadata)
    return ApprovalRecord(request=req, status="pending")


def test_headers_match_case_insensitively():
    info = request(**{"X-Run-Id": "r1"})
    assert info.header("x-run-id") == "r1"
    assert info.header("missing") is None


@pytest.mark.parametrize(
    ("value", "token"),
    [
        ("Bearer abc", "abc"),
        ("bearer  abc ", "abc"),
        ("BEARER abc", "abc"),
        ("Basic abc", ""),
        ("Bearer", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_bearer_token_parsing(value: str | None, token: str):
    assert bearer_token(request(value)) == token


async def test_a_token_table_maps_tokens_to_principals():
    auth = bearer_auth({"t-alice": ALICE})
    assert await auth(request("Bearer t-alice")) is ALICE
    assert await auth(request("Bearer t-bob")) is None
    assert await auth(request()) is None


async def test_a_verifier_callable_is_awaited():
    seen: list[str] = []

    async def verify(token: str) -> Principal | None:
        seen.append(token)
        return ALICE if token == "ok" else None

    auth = bearer_auth(verify)
    assert await auth(request("Bearer ok")) is ALICE
    assert await auth(request("Bearer no")) is None
    assert await auth(request("Basic ok")) is None
    assert seen == ["ok", "no"]  # no verifier call without a bearer token


async def test_a_verifier_returning_another_type_fails_closed():
    async def verify(token: str) -> Any:
        return "alice"

    with pytest.raises(ToolUsageError, match="not Principal"):
        await bearer_auth(verify)(request("Bearer x"))


@pytest.mark.parametrize(
    "tokens",
    [{"": ALICE}, {"  ": ALICE}, {"t": "alice"}, {1: ALICE}, 42],
)
def test_bad_token_tables_are_refused(tokens: Any):
    with pytest.raises(ToolUsageError):
        bearer_auth(tokens)


def test_principal_repr_hides_claims():
    assert "secret" not in repr(Principal("a", claims={"email": "secret"}))


async def test_default_authorize_allows_every_authenticated_principal():
    reg = Registry()

    @reg.tool
    def ping() -> str:
        """Ping."""
        return "pong"

    assert await allow_authenticated(ALICE, reg.get("ping"), None) is True


async def test_default_decision_rule_is_invoker_or_approver():
    mine = record(principal="alice")
    assert await owner_or_approver(ALICE, mine) is True
    assert await owner_or_approver(Principal("bob"), mine) is False
    assert await owner_or_approver(Principal("bob", frozenset({APPROVER_SCOPE})), mine) is True
    assert await owner_or_approver(ALICE, record()) is False  # no invoker recorded: approvers only
    assert await owner_or_approver(Principal(""), record(principal=7)) is False


def test_principal_ref_hashes_ids():
    assert principal_ref("alice@example.com") != "alice@example.com"
    assert len(principal_ref("alice@example.com")) == 12
    assert principal_ref(None) == principal_ref("") == "anonymous"


def test_invoker_of_rebuilds_the_principal_or_refuses_malformed_metadata():
    from agent_adapter.server.auth import invoker_of

    rebuilt = invoker_of(record(principal="alice", principal_scopes=["b", "a"]))
    assert rebuilt == Principal("alice", scopes=frozenset({"a", "b"}))
    assert invoker_of(record(anonymous=True)) == Principal("")
    assert invoker_of(record()) is None
    assert invoker_of(record(principal="alice", principal_scopes="a")) is None
    assert invoker_of(record(principal="alice", principal_scopes=[1])) is None
    assert invoker_of(record(principal="alice")) == Principal("alice")
