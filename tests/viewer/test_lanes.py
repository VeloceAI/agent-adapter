"""Timeline spans and lanes: parallel calls side by side, retries, timeouts, cache, compression."""

from __future__ import annotations

from typing import Any

from agent_adapter.viewer import assign_lanes, build_spans
from tests.viewer.helpers import as_dict, event


def events(*items: Any) -> list[dict[str, Any]]:
    return [as_dict(item) if not isinstance(item, tuple) else as_dict(*item) for item in items]


def by_name(spans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {span["name"]: span for span in spans}


def test_parallel_calls_get_separate_lanes_and_sequential_ones_share() -> None:
    spans = build_spans(
        events(
            event("tool.started", 1.0, "a", tool="a", kind=""),
            event("tool.started", 1.1, "b", tool="b", kind=""),
            event("tool.finished", 1.5, "a", tool="a", latency_ms=500, attempts=1, cached=False),
            event("tool.finished", 1.6, "b", tool="b", latency_ms=500, attempts=1, cached=False),
            event("tool.started", 2.0, "c", tool="c", kind=""),
            event("tool.finished", 2.2, "c", tool="c", latency_ms=200, attempts=1, cached=False),
        )
    )
    lanes = {span["name"]: span["lane"] for span in spans}
    assert lanes == {"a": 0, "b": 1, "c": 0}
    assert by_name(spans)["a"]["status"] == "ok"


def test_assign_lanes_counts_lanes_and_touching_spans_share() -> None:
    spans = [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}, {"start": 0.5, "end": 1.5}]
    assert assign_lanes(spans) == 2
    assert [span["lane"] for span in spans] == [0, 0, 1]
    assert assign_lanes([]) == 0


def test_spans_show_retries_timeouts_errors_and_cache_hits() -> None:
    spans = by_name(
        build_spans(
            events(
                event("tool.started", 1.0, "r", tool="slow", kind=""),
                event("tool.retry", 1.1, "r", tool="slow", attempt=1, delay_s=0.1, error_kind="x"),
                event(
                    "tool.failed",
                    2.0,
                    "r",
                    tool="slow",
                    error_kind="tool-timeout",
                    attempts=2,
                    latency_ms=1000,
                ),
                event("tool.started", 3.0, "k", tool="cached", kind=""),
                event("tool.cached", 3.1, "k", tool="cached"),
                event("llm.started", 4.0, "m", provider="p", model="m", stage="s", stream=False),
                event(
                    "llm.finished",
                    4.5,
                    "m",
                    provider="p",
                    model="m",
                    stage="s",
                    latency_ms=500,
                    usage={},
                    cost_usd=0.25,
                    finish_reason="stop",
                ),
            )
        )
    )
    slow = spans["slow"]
    assert (slow["retries"], slow["timeout"], slow["error_kind"], slow["status"]) == (
        1,
        True,
        "tool-timeout",
        "error",
    )
    assert slow["marks"][0]["type"] == "tool.retry"
    assert spans["cached"]["cached"] and spans["cached"]["status"] == "cached"
    assert spans["p/m"]["cost_usd"] == 0.25


def test_compression_is_a_span_ending_at_its_event_with_tokens_before_and_after() -> None:
    (span,) = build_spans(
        events(
            event(
                "context.compressed",
                10.0,
                "cmp_1",
                provider="p",
                model="m",
                trigger="threshold",
                latency_ms=500,
                tokens_before=900,
                tokens_after=300,
                target_tokens=400,
                messages_before=9,
                messages_after=3,
                steps=["trim"],
                summary_cost_usd=0.01,
            )
        )
    )
    assert span["family"] == "context"
    assert (span["start"], span["end"]) == (9.5, 10.0)
    assert span["tokens"] == {"before": 900, "after": 300}
    assert span["cost_usd"] == 0.01


def test_calls_inherit_the_flow_node_they_ran_under() -> None:
    spans = build_spans(
        events(
            event("node.started", 1.0, "n1", flow="f", node="search", visit=1),
            (event("tool.started", 1.1, "t1", tool="lookup", kind=""), "n1"),
            (event("approval.requested", 1.2, "t1", tool="lookup", approval_id="a"), "n1"),
            (event("tool.pending", 1.3, "t1", tool="lookup", approval_id="a"), "n1"),
            event("node.paused", 1.4, "n1", flow="f", node="search", approval_id="a"),
        )
    )
    tool = by_name(spans)["lookup"]
    assert tool["node"] == "search"
    assert tool["status"] == "pending"
    assert [mark["type"] for mark in tool["marks"]] == ["approval.requested"]
    assert by_name(spans)["search"]["status"] == "paused"


def test_open_spans_run_to_the_last_event_and_orphan_ends_still_show() -> None:
    spans = by_name(
        build_spans(
            events(
                event("flow.started", 1.0, flow="f"),
                event("node.finished", 3.0, flow="f", node="n", visit=1),
            )
        )
    )
    assert spans["f"]["status"] == "running" and spans["f"]["end"] == 3.0
    assert spans["n"]["status"] == "ok" and spans["n"]["start"] == 3.0


def test_unknown_families_and_events_without_ids_are_handled() -> None:
    spans = build_spans(
        events(
            event("custom.note", 1.0, text="x"),
            event("run.started", 1.0),
            event("embedding.started", 1.0, provider="p", model="e", stage="s", texts=2),
            event(
                "rerank.failed",
                2.0,
                provider="p",
                model="r",
                stage="s",
                texts=2,
                error_kind="boom",
                latency_ms=5,
            ),
        )
    )
    assert [(span["family"], span["name"], span["status"]) for span in spans] == [
        ("embedding", "p/e", "running"),
        ("rerank", "p/r", "error"),
    ]
