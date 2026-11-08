"""MemoryScope, MemoryRecord and the shared input checks."""

import math
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_adapter.errors import ConfigError, ToolUsageError
from agent_adapter.memory import MemoryHit, MemoryScope
from agent_adapter.memory.records import (
    MAX_BATCH,
    MAX_K,
    check_filters,
    check_ids,
    check_query,
    check_records,
    check_retention,
    check_scope,
    matches,
)
from tests.memory.helpers import ALICE, ALICE_ALL, ALICE_S2, BOB, days_ago, rec


def test_scope_needs_at_least_one_id():
    with pytest.raises(ValidationError):
        MemoryScope()
    with pytest.raises(ValidationError):
        MemoryScope(user_id="x" * 257)
    with pytest.raises(ValidationError):
        MemoryScope(user_id="u", tenant="t")


def test_user_scope_covers_all_its_sessions_and_never_another_user():
    assert ALICE_ALL.covers(ALICE) and ALICE_ALL.covers(ALICE_S2)
    assert ALICE.covers(ALICE) and not ALICE.covers(ALICE_S2)
    assert not ALICE_ALL.covers(BOB)
    assert not ALICE.covers(ALICE_ALL)  # a session query never sees user-level records


def test_an_empty_user_is_its_own_partition():
    anon = MemoryScope(session_id="s1")
    assert anon.covers(MemoryScope(session_id="s1"))
    assert not anon.covers(ALICE)  # same session id, but a user's record
    agent = MemoryScope(agent_id="helper")
    assert agent.covers(MemoryScope(agent_id="helper", session_id="x"))
    assert not agent.covers(MemoryScope(user_id="alice", agent_id="helper"))


def test_record_checks_text_metadata_and_times():
    with pytest.raises(ValidationError):
        rec(ALICE, "")
    with pytest.raises(ValidationError):
        rec(ALICE, "x" * 20_001)
    with pytest.raises(ValidationError):
        rec(ALICE, "t", metadata={"bad": math.nan})
    with pytest.raises(ValidationError):
        rec(ALICE, "t", metadata={"obj": object()})
    with pytest.raises(ValidationError):
        rec(ALICE, "t", metadata={"big": "x" * 16_001})
    naive = rec(ALICE, "t", created_at=datetime(2026, 1, 1))
    assert naive.created_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert MemoryHit(record=naive).score == 0.0


def test_live_honours_expiry_and_retention():
    now = datetime.now(UTC)
    assert rec(ALICE, "t").live(now, None)
    assert not rec(ALICE, "t", expires_at=days_ago(1)).live(now, None)
    assert rec(ALICE, "t", created_at=days_ago(5)).live(now, 10)
    assert not rec(ALICE, "t", created_at=days_ago(11)).live(now, 10)


@pytest.mark.parametrize("bad", [0, -1, math.inf, math.nan, True, "7"])
def test_retention_must_be_a_positive_number(bad):
    with pytest.raises(ConfigError):
        check_retention(bad)


def test_retention_none_and_numbers_pass():
    assert check_retention(None) is None
    assert check_retention(7) == 7.0


def test_scope_and_record_and_id_checks():
    with pytest.raises(ToolUsageError):
        check_scope({"user_id": "alice"})
    assert check_records(rec(ALICE, "one"))[0].text == "one"
    with pytest.raises(ToolUsageError):
        check_records("text")
    with pytest.raises(ToolUsageError):
        check_records([{"text": "x"}])
    with pytest.raises(ToolUsageError):
        check_records([rec(ALICE, "a", id="m1"), rec(ALICE, "b", id="m1")])
    with pytest.raises(ToolUsageError):
        check_records([rec(ALICE, "a")] * (MAX_BATCH + 1))
    with pytest.raises(ToolUsageError):
        check_ids("m1")
    with pytest.raises(ToolUsageError):
        check_ids([""])
    with pytest.raises(ToolUsageError):
        check_ids(["m"] * (MAX_BATCH + 1))


@pytest.mark.parametrize(
    "query, k", [(1, 5), ("q", 0), ("q", MAX_K + 1), ("q", True), ("x" * 20_001, 1)]
)
def test_query_checks(query, k):
    with pytest.raises(ToolUsageError):
        check_query(query, k)


@pytest.mark.parametrize(
    "filters", [["a"], {"bad key": 1}, {'q"uote': 1}, {"k": None}, {"k": math.inf}, {"k": [1]}]
)
def test_filter_checks(filters):
    with pytest.raises(ToolUsageError):
        check_filters(filters)


def test_matches_keeps_bools_apart_from_numbers():
    record = rec(ALICE, "t", metadata={"n": 1, "flag": True, "kind": "note"})
    assert matches(record, {"n": 1, "kind": "note"})
    assert matches(record, {"n": 1.0})
    assert not matches(record, {"n": True})
    assert not matches(record, {"flag": 1})
    assert not matches(record, {"missing": "x"})
    assert check_filters(None) == {}
