"""The lenient parser of streaming JSON text and its throttled tracker."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.core._partial_json import MAX_TEXT, PartialJSON, parse_partial_json


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("{", {}),
        ('{"a', {}),
        ('{"a":', {}),
        ('{"a": 1', {}),
        ('{"a": 1,', {"a": 1}),
        ('{"a": "he', {"a": "he"}),
        ('{"a": "he\\', {"a": "he"}),
        ('{"a": "x\\u00', {"a": "x"}),
        ('{"a": "x\\u00e9', {"a": "xé"}),
        ('{"a": "\\n', {"a": "\n"}),
        ('{"a": [1, 2', {"a": [1]}),
        ('{"a": [1, 2]}', {"a": [1, 2]}),
        ('{"a": tru', {}),
        ('{"a": true}', {"a": True}),
        ('[{"b": nul', [{}]),
        ('{"a": {"b": "c"}, "d"', {"a": {"b": "c"}}),
        ('  "abc', "abc"),
        ("true ", True),
        ('{"a": -1.5e3}', {"a": -1500.0}),
    ],
)
def test_parse_partial_json_keeps_what_is_complete(text: str, value: Any) -> None:
    assert parse_partial_json(text) == (True, value)


@pytest.mark.parametrize("text", ["", "   ", "tr", "12", "nope ", '{"a" 1}', "{1: 2}", "[1 x]"])
def test_parse_partial_json_without_a_value(text: str) -> None:
    assert parse_partial_json(text) == (False, None)


def test_parse_partial_json_refuses_deep_nesting() -> None:
    assert parse_partial_json("[" * 100) == (False, None)


def test_tracker_reports_only_changes() -> None:
    tracker = PartialJSON()
    assert tracker.feed('{"a') == (True, {})
    assert tracker.feed('"') == (False, None)  # still {}
    assert tracker.feed(': "x') == (True, {"a": "x"})
    assert tracker.feed("") == (False, None)
    assert tracker.feed('"}') == (False, None)  # same value, now complete
    assert tracker.feed("  ") == (False, None)


def test_tracker_first_chunk_with_a_brace() -> None:
    assert PartialJSON().feed("{") == (True, {})


def test_tracker_throttles_long_text_and_stops_at_the_cap() -> None:
    tracker = PartialJSON()
    tracker.feed('{"a": "' + "x" * 2000)
    assert tracker.feed("y") == (False, None)  # under 1/8 more text: not parsed again
    changed, value = tracker.feed("y" * 400)
    assert changed and value["a"].endswith("y" * 401)
    tracker.feed("z" * MAX_TEXT)
    assert len(tracker.text) == MAX_TEXT
    assert tracker.feed("more") == (False, None)


def test_an_open_array_and_a_bad_escape() -> None:
    assert parse_partial_json("[") == (True, [])
    assert parse_partial_json(r'{"a": "x\q') == (True, {"a": ""})  # invalid escape: no prefix


@pytest.mark.parametrize(
    ("text", "value"),
    [
        (r'{"p": "C:\\', "C:\\"),  # an escaped backslash: complete
        (r'{"p": "C:\\\\', "C:\\\\"),
        ('{"p": "C:' + "\\" * 5, "C:" + "\\" * 2),  # odd run: the last escape is cut
        (r'{"p": "a\"b\\', 'a"b\\'),
    ],
)
def test_trailing_backslash_runs_keep_complete_escapes(text: str, value: str) -> None:
    assert parse_partial_json(text) == (True, {"p": value})
