"""tests.work: line counts are deterministic, capped, and reported from a child interpreter."""

import pytest

from agent_adapter.schema.models import _field_names
from tests.work import WorkExceeded, counting, run_child

CHILD = """
import json
from tests.work import counting
from agent_adapter.schema.models import _field_names
with counting(10**6) as work:
    _field_names(["a-b", "a.b", "a b"])
print(json.dumps({"lines": work.lines}))
"""


def test_counting_counts_agent_adapter_lines_the_same_every_run():
    with counting(10**6) as first:
        _field_names(["a-b", "a.b", "a b"])
    with counting(10**6) as second:
        _field_names(["a-b", "a.b", "a b"])
    assert 0 < first.lines == second.lines


def test_counting_ignores_lines_outside_the_package():
    with counting(10**6) as work:
        sum(len(str(i)) for i in range(1000))
    assert work.lines == 0


def test_counting_raises_past_its_cap():
    with pytest.raises(WorkExceeded, match="more than 5 lines"), counting(5):
        _field_names([f"x-{i}" for i in range(100)])


def test_run_child_returns_the_last_json_line():
    assert run_child(CHILD)["lines"] > 0


def test_run_child_fails_with_the_child_error():
    with pytest.raises(AssertionError, match="ValueError: boom"):
        run_child("raise ValueError('boom')")
