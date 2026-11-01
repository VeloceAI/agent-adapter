import json
import logging
import threading

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import LedgerEntry
from agent_adapter.cost import CallbackSink, JsonlSink, LedgerSink, MemorySink, read_jsonl


def entry(call_id="c1", run_id="r1", **fields):
    return LedgerEntry(run_id=run_id, call_id=call_id, kind="llm", **fields)


def test_sinks_satisfy_the_protocol(tmp_path):
    for sink in (MemorySink(), CallbackSink(print), JsonlSink(tmp_path / "x.jsonl")):
        assert isinstance(sink, LedgerSink)


def test_memory_sink_appends():
    sink = MemorySink()
    sink.write(entry("a"))
    sink.write(entry("a"))
    assert [e.call_id for e in sink.entries] == ["a", "a"]


def test_callback_sink_calls_the_function():
    seen = []
    sink = CallbackSink(seen.append)
    sink.write(entry())
    assert seen == [entry()]


def test_jsonl_round_trip_keeps_every_field(tmp_path):
    path = tmp_path / "ledger.jsonl"
    sink = JsonlSink(path)
    original = entry(
        cache_write_by_ttl={"5m": 3},
        cost_usd=0.25,
        metadata={"tenant": "acme"},
        thread_id="t1",
    )
    sink.write(original)
    sink.write(entry("c2", cost_usd=None))
    assert read_jsonl(path) == [original, entry("c2")]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["cost_usd"] == 0.25


def test_jsonl_appends_to_an_existing_file_as_utf8(tmp_path):
    path = tmp_path / "ledger.jsonl"
    JsonlSink(path).write(entry(stage="résumé"))
    JsonlSink(str(path)).write(entry("c2"))
    assert [e.stage for e in read_jsonl(path)] == ["résumé", ""]
    assert "résumé" in path.read_bytes().decode("utf-8")


def test_read_jsonl_last_write_per_key_wins_in_first_seen_order(tmp_path):
    path = tmp_path / "ledger.jsonl"
    sink = JsonlSink(path)
    sink.write(entry("a", cost_usd=None))
    sink.write(entry("b"))
    sink.write(entry("a", cost_usd=1.0))
    sink.write(entry("a", run_id="r2"))
    result = read_jsonl(path)
    assert [(e.run_id, e.call_id, e.cost_usd) for e in result] == [
        ("r1", "a", 1.0),
        ("r1", "b", None),
        ("r2", "a", None),
    ]


def test_read_jsonl_skips_blank_lines_and_a_torn_last_line(tmp_path, caplog):
    path = tmp_path / "ledger.jsonl"
    path.write_text(entry().model_dump_json() + "\n\n" + '{"run_id": "r1", "ca', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.sinks"):
        assert read_jsonl(path) == [entry()]
    assert "incomplete last line 3" in caplog.text


def test_read_jsonl_rejects_a_bad_line_in_the_middle(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("not json\n" + entry().model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="line 1"):
        read_jsonl(path)


def test_read_jsonl_rejects_an_invalid_entry(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"run_id": "r"}\n{"run_id": "r"}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="line 1"):
        read_jsonl(path)


def test_read_jsonl_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read ledger file"):
        read_jsonl(tmp_path / "absent.jsonl")


def test_jsonl_write_failure_propagates_to_the_ledger(tmp_path):
    sink = JsonlSink(tmp_path / "missing-dir" / "ledger.jsonl")
    with pytest.raises(OSError):
        sink.write(entry())


def test_jsonl_concurrent_writes_keep_whole_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    sink = JsonlSink(path)

    def work(n):
        for i in range(50):
            sink.write(entry(f"{n}-{i}", metadata={"pad": "x" * 200}))

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(read_jsonl(path)) == 200
