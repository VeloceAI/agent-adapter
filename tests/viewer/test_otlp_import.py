"""OTLP JSON import: GenAI spans become a viewer run; bad files fail closed; nothing runs."""

from __future__ import annotations

import io
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.cli.main import main
from agent_adapter.core.errors import ConfigError
from agent_adapter.viewer import _otlp_import as otlp_import
from agent_adapter.viewer import import_otlp
from agent_adapter.viewer._otlp_import import attributes
from tests.viewer.helpers import lines, refreshed

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "viewer_search_import.py"
NS = 1_000_000_000
Span = dict[str, Any]


def a(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, list):
        return {
            "key": key,
            "value": {"arrayValue": {"values": [{"stringValue": v} for v in value]}},
        }
    return {"key": key, "value": {"stringValue": value}}


def span(sid: str, name: str, start: float, end: float, *attrs: Any, **extra: Any) -> Span:
    return {
        "traceId": extra.pop("trace", "t1"),
        "spanId": sid,
        "parentSpanId": extra.pop("parent", ""),
        "name": name,
        "startTimeUnixNano": str(int(start * NS)),
        "endTimeUnixNano": str(int(end * NS)),
        "attributes": list(attrs),
        **extra,
    }


def doc(*spans: Span) -> dict[str, Any]:
    return {"resourceSpans": [{"resource": {}, "scopeSpans": [{"spans": list(spans)}]}]}


TRACE = doc(
    span(
        "f1", "invoke_workflow shop", 1.0, 5.0,
        a("gen_ai.operation.name", "invoke_workflow"), a("gen_ai.workflow.name", "shop"),
        a("agent_adapter.flow.status", "done"), a("agent_adapter.flow.steps", 2),
    ),
    span(
        "n1", "node shop.plan", 1.1, 4.0, a("agent_adapter.flow.node", "plan"),
        a("agent_adapter.flow.visit", 1), parent="f1",
    ),
    span(
        "l1", "generate_content shop-model", 1.2, 2.0,
        a("gen_ai.operation.name", "generate_content"), a("gen_ai.provider.name", "gcp.gemini"),
        a("gen_ai.request.model", "shop-model"), a("gen_ai.usage.input_tokens", 100),
        a("gen_ai.usage.cache_read.input_tokens", 30), a("gen_ai.usage.output_tokens", 7),
        a("gen_ai.response.finish_reasons", ["stop"]), a("agent_adapter.cost_usd", 0.25),
        a("gen_ai.input.messages", '[{"role": "user", "content": "mugs?", "api_key": "sk-1"}]'),
        parent="n1",
    ),
    span(
        "t1s", "execute_tool stock", 2.1, 2.3, a("gen_ai.operation.name", "execute_tool"),
        a("gen_ai.tool.name", "stock"), a("gen_ai.tool.call.arguments", '{"item": "mug"}'),
        a("agent_adapter.cache_hit", True), parent="n1",
    ),
    span(
        "t2s", "execute_tool refund", 2.4, 2.6, a("gen_ai.operation.name", "execute_tool"),
        a("gen_ai.tool.name", "refund"), a("error.type", "tool-timeout"), parent="n1",
        status={"code": 2},
    ),
    span("e1", "embeddings shop-embed", 3.0, 3.1, a("gen_ai.operation.name", "embeddings")),
    span("x1", "GET /health", 3.2, 3.3, a("http.method", "GET")),
)  # fmt: skip


def write(tmp_path: Path, document: Any) -> Path:
    path = tmp_path / "traces.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_genai_spans_become_one_run_with_events(tmp_path: Path) -> None:
    keys = import_otlp(write(tmp_path, TRACE), tmp_path / "runs")
    assert keys == ["otlp-t1"]
    (summary,) = refreshed(tmp_path / "runs").summaries()
    assert summary["run_id"] == "otlp-t1" and summary["framework"] == "otlp-import"
    assert summary["flows"] == ["shop"] and summary["errors"] == 1
    assert summary["cost_usd"] == 0.25
    (path,) = (tmp_path / "runs").glob("*.jsonl")
    records = lines(path)
    assert records[0]["kind"] == "meta" and records[0]["capture_content"] is True
    events = {(r["type"], r["call_id"]): r for r in records[1:]}
    llm = events[("llm.finished", "l1")]["data"]
    assert llm["provider"] == "gemini" and llm["model"] == "shop-model"
    assert llm["usage"]["input_tokens"] == 70 and llm["usage"]["cached_input_tokens"] == 30
    assert llm["finish_reason"] == "stop" and llm["latency_ms"] == pytest.approx(800.0)
    started = events[("llm.started", "l1")]["data"]["input_messages"]
    assert started[0]["content"] == "mugs?" and started[0]["api_key"] != "sk-1"  # masked
    assert events[("tool.started", "t1s")]["data"]["arguments"] == {"item": "mug"}
    assert events[("tool.finished", "t1s")]["data"]["cached"] is True
    assert events[("tool.failed", "t2s")]["data"]["error_kind"] == "tool-timeout"
    assert events[("node.finished", "n1")]["data"] == {"flow": "shop", "node": "plan", "visit": 1}
    assert events[("tool.started", "t1s")]["parent_id"] == "n1"
    assert ("embedding.finished", "e1") in events
    assert not any(call == "x1" for _, call in events)
    detail = refreshed(tmp_path / "runs").detail("otlp-t1")
    assert detail is not None
    assert {s["family"] for s in detail["spans"]} >= {"flow", "node", "llm", "tool", "embedding"}


def test_runs_split_by_conversation_id_and_reimport_rewrites(tmp_path: Path) -> None:
    document = doc(
        span("a", "chat m", 1.0, 2.0, a("gen_ai.operation.name", "chat"),
             a("gen_ai.conversation.id", "run_9")),
        span("b", "chat m", 3.0, 4.0, a("gen_ai.operation.name", "chat"), trace="t2"),
    )  # fmt: skip
    source = write(tmp_path, document)
    assert sorted(import_otlp(source, tmp_path / "runs")) == ["otlp-run_9", "otlp-t2"]
    assert sorted(import_otlp(source, tmp_path / "runs")) == ["otlp-run_9", "otlp-t2"]
    assert len(list((tmp_path / "runs").glob("*.jsonl"))) == 2
    records = lines(next((tmp_path / "runs").glob("*otlp-t2*")))
    assert records[0]["capture_content"] is False


def test_failed_llm_and_flow_spans(tmp_path: Path) -> None:
    document = doc(
        span("l", "chat m", 1.0, 2.0, a("gen_ai.operation.name", "chat"),
             status={"code": "STATUS_CODE_ERROR"}),
        span("f", "invoke_workflow w", 1.0, 3.0, a("gen_ai.operation.name", "invoke_workflow"),
             status={"code": 2}),
        span("e", "embeddings m", 1.0, 1.5, a("gen_ai.operation.name", "embeddings"),
             a("error.type", "x")),
        span("n", "node w.step", 2.0, 1.0, a("error.type", "boom")),
    )  # fmt: skip
    import_otlp(write(tmp_path, document), tmp_path / "runs")
    records = lines(next((tmp_path / "runs").glob("*.jsonl")))
    by_type = {r["type"]: r for r in records[1:]}
    assert by_type["llm.failed"]["data"]["error_kind"] == "error"
    assert by_type["flow.finished"]["data"]["status"] == "failed"
    assert by_type["embedding.failed"]["data"]["error_kind"] == "x"
    assert by_type["node.failed"]["ts"] == by_type["node.started"]["ts"]  # end before start


@pytest.mark.parametrize(
    "document",
    [[], {"resourceSpans": "x"}, {"other": 1}, "text"],
)
def test_not_an_otlp_export_is_refused(tmp_path: Path, document: Any) -> None:
    with pytest.raises(ConfigError, match="OTLP"):
        import_otlp(write(tmp_path, document), tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_bad_json_too_big_and_too_many_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(ConfigError, match="not JSON"):
        import_otlp(bad, tmp_path / "runs")
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 100_000, encoding="utf-8")
    with pytest.raises(ConfigError, match="not JSON"):
        import_otlp(deep, tmp_path / "runs")
    monkeypatch.setattr(otlp_import, "MAX_IMPORT_BYTES", 10)
    with pytest.raises(ConfigError, match="MB"):
        import_otlp(write(tmp_path, TRACE), tmp_path / "runs")
    monkeypatch.setattr(otlp_import, "MAX_IMPORT_BYTES", 10**9)
    monkeypatch.setattr(otlp_import, "MAX_SPANS", 2)
    with pytest.raises(ConfigError, match="spans"):
        import_otlp(write(tmp_path, TRACE), tmp_path / "runs")
    monkeypatch.setattr(otlp_import, "MAX_SPANS", 100)
    monkeypatch.setattr(otlp_import, "MAX_RUNS", 1)
    two = doc(
        span("a", "chat m", 1.0, 2.0, a("gen_ai.operation.name", "chat")),
        span("b", "chat m", 1.0, 2.0, a("gen_ai.operation.name", "chat"), trace="t2"),
    )
    with pytest.raises(ConfigError, match="traces"):
        import_otlp(write(tmp_path, two), tmp_path / "runs")


def test_malformed_parts_are_skipped(tmp_path: Path) -> None:
    document = {
        "resourceSpans": [
            "x",
            {"scopeSpans": "x"},
            {"instrumentationLibrarySpans": [{"spans": ["x", span("a", "chat m", 1, 2,
                a("gen_ai.operation.name", "chat"), a("gen_ai.usage.input_tokens", "many"))]}]},
            {"scopeSpans": ["x", {"spans": "x"}]},
        ]
    }  # fmt: skip
    assert import_otlp(write(tmp_path, document), tmp_path / "runs") == ["otlp-t1"]
    assert import_otlp(write(tmp_path, doc()), tmp_path / "empty") == []


def test_attribute_values() -> None:
    items = [
        a("s", "x"), a("b", False), a("i", 3), a("d", 1.5), a("l", ["p", "q"]),
        {"key": "bad_int", "value": {"intValue": "x"}},
        {"key": "bad_double", "value": {"doubleValue": "x"}},
        {"key": "none", "value": {}}, {"key": "nested", "value": "x"}, {"no": "key"}, "x",
    ]  # fmt: skip
    assert attributes(items) == {
        "s": "x", "b": False, "i": 3, "d": 1.5, "l": ["p", "q"], "bad_int": None,
        "bad_double": None, "none": None, "nested": None,
    }  # fmt: skip
    assert attributes("x") == {}


def test_cli_import_then_export(tmp_path: Path) -> None:
    out = io.StringIO()
    source = write(tmp_path, TRACE)
    runs, report = tmp_path / "runs", tmp_path / "r.html"
    code = main(["ui", str(runs), "--import", str(source), "--export", str(report)], out)
    assert code == 0
    assert out.getvalue().splitlines()[0] == f"imported 1 run(s) from {source} into {runs}"
    assert report.is_file()


def test_cli_import_refuses_a_run_file_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = write(tmp_path, TRACE)
    import_otlp(source, tmp_path / "runs")
    (segment,) = (tmp_path / "runs").glob("*.jsonl")
    code = main(["ui", str(segment), "--import", str(source), "--export", "x.html"], io.StringIO())
    assert code == 2 and "runs folder" in capsys.readouterr().err


def test_the_example_imports_a_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module: dict[str, Any] = runpy.run_path(str(EXAMPLE))
    assert module["main"](tmp_path) == ["otlp-t1 ok 4 events otlp-import"]
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out == "otlp-t1 ok 4 events otlp-import\n"


def test_bad_timestamps_count_as_zero(tmp_path: Path) -> None:
    bad = span("a", "chat m", 1.0, 2.0, a("gen_ai.operation.name", "chat"))
    bad["startTimeUnixNano"], bad["endTimeUnixNano"] = "soon", None
    import_otlp(write(tmp_path, doc(bad)), tmp_path / "runs")
    records = lines(next((tmp_path / "runs").glob("*.jsonl")))
    assert [r["ts"] for r in records[1:]] == [0.0, 0.0]


@pytest.mark.parametrize(
    "start", [float("inf"), float("-inf"), -5 * NS, 10**20, "1e400", 2**70], ids=str
)
def test_out_of_range_times_count_as_zero_and_the_run_is_listed(tmp_path: Path, start: Any) -> None:
    bad = span("a", "chat m", 1.0, 2.0, a("gen_ai.operation.name", "chat"))
    bad["startTimeUnixNano"] = start
    bad["endTimeUnixNano"] = start
    assert import_otlp(write(tmp_path, doc(bad)), tmp_path / "runs") == ["otlp-t1"]
    (summary,) = refreshed(tmp_path / "runs").summaries()  # the file name parses
    assert summary["run_id"] == "otlp-t1"


def test_long_ids_are_cut(tmp_path: Path) -> None:
    long = span("s" * 200_000, "chat m", 1.0, 2.0, a("gen_ai.operation.name", "chat"),
                parent="p" * 1_000, trace="t" * 1_000)  # fmt: skip
    import_otlp(write(tmp_path, doc(long)), tmp_path / "runs")
    (path,) = (tmp_path / "runs").glob("*.jsonl")
    records = lines(path)
    assert len(records[1]["call_id"]) == 128 and len(records[1]["parent_id"]) == 128
    assert records[0]["run_id"] == "otlp-" + "t" * 128
    assert path.stat().st_size < 10_000


@pytest.mark.parametrize(
    "extra",
    [["--run", "X"], ["--approvals", "a.db"], ["--export", "r.html", "--approvals", "a.db"]],
    ids=["run-without-export", "approvals-without-reviewer", "export-with-approvals"],
)
def test_cli_checks_every_option_before_importing(
    tmp_path: Path, extra: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    source = write(tmp_path, TRACE)
    runs = tmp_path / "runs"
    code = main(["ui", str(runs), "--import", str(source), *extra], io.StringIO())
    assert code == 2 and capsys.readouterr().err
    assert not runs.exists()


def test_cli_import_into_a_served_folder_closes_the_server_on_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    code = main(["ui", str(tmp_path / "runs"), "--port", "0", "--import", str(bad)], io.StringIO())
    assert code == 2 and "not JSON" in capsys.readouterr().err
