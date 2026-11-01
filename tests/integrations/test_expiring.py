"""`ExpiringTable`: the TTL, bound and lookup rules both token tables share."""

import logging

from agent_adapter.integrations._expiring import ExpiringTable
from tests.fakes.clock import FakeClock


def table(**options):
    clock = FakeClock()
    base = {"ttl": 10.0, "max_entries": 2, "clock": clock, "label": "things"}
    return ExpiringTable(**{**base, **options}), clock


def test_entries_expire_at_their_deadline(caplog):
    entries, clock = table()
    entries.add("a", 1)
    entries.add("b", 2, expires_at=clock() + 20.0)
    clock.advance(10.0)
    with caplog.at_level(logging.DEBUG):
        assert len(entries) == 1 and entries.pop("a") is None
    assert "1 things expired" in caplog.text
    assert entries.pop("b") == 2 and entries.pop("b") is None


def test_a_full_table_returns_what_it_dropped_oldest_first():
    entries, _ = table()
    assert entries.add("a", 1) == [] and entries.add("b", 2) == []
    assert entries.add("c", 3) == [1]
    assert entries.pop("a") is None and len(entries) == 2


def test_pop_first_takes_the_oldest_match_and_clear_counts():
    entries, _ = table(max_entries=5)
    for key, value in (("a", 1), ("b", 2), ("c", 3)):
        entries.add(key, value)
    assert entries.pop_first(lambda v: v > 1) == 2
    assert entries.pop_first(lambda v: v > 5) is None
    assert entries.clear() == 2 and len(entries) == 0
    assert entries.deadline() == 10.0
