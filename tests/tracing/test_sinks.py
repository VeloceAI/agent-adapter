import logging
import threading

import pytest

from agent_adapter.core.ports import EventSink
from agent_adapter.core.types import Event
from agent_adapter.tracing import CallbackEventSink, EventLog, FanoutSink, LoggingSink
from tests.fakes.sinks import RecordingSink


def ev(type="tool.started", **data):
    return Event(type, run_id="r1", call_id="c1", data=data)


def test_every_sink_satisfies_the_port():
    for sink in (EventLog(), LoggingSink(), CallbackEventSink(print), FanoutSink()):
        assert isinstance(sink, EventSink)


def test_event_log_keeps_events_in_order_and_filters_by_prefix():
    log = EventLog()
    for t in ("tool.started", "tool.finished", "llm.started"):
        log.emit(ev(t))
    assert [e.type for e in log.events] == ["tool.started", "tool.finished", "llm.started"]
    assert [e.type for e in log.of_type("tool.")] == ["tool.started", "tool.finished"]
    assert log.of_type("node.") == []


def test_event_log_is_bounded_dropping_the_oldest():
    log = EventLog(max_events=3)
    for i in range(5):
        log.emit(ev("custom.n", i=i))
    assert [e.data["i"] for e in log.events] == [2, 3, 4]


def test_event_log_rejects_a_non_positive_bound():
    from agent_adapter.core.errors import ConfigError

    with pytest.raises(ConfigError, match="max_events"):
        EventLog(max_events=0)


def test_event_log_is_thread_safe():
    log = EventLog(max_events=1000)

    def work():
        for _ in range(200):
            log.emit(ev())

    threads = [threading.Thread(target=work) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(log.events) == 1000


def test_logging_sink_logs_type_ids_and_data_at_debug_without_content(caplog):
    logger = logging.getLogger("test.events")
    with caplog.at_level(logging.DEBUG, logger="test.events"):
        LoggingSink(logger).emit(ev(tool="search", arguments={"q": "secret"}))
    [record] = caplog.records
    assert record.levelno == logging.DEBUG
    message = record.getMessage()
    assert "tool.started" in message and "run=r1" in message and "call=c1" in message
    assert "search" in message and "secret" not in message
    assert "omitted=['arguments']" in message


def test_logging_sink_omits_flow_state_values_without_content(caplog):
    logger = logging.getLogger("test.state")
    with caplog.at_level(logging.DEBUG, logger="test.state"):
        LoggingSink(logger).emit(ev(changed=["x"], updates={"x": "private"}))
    message = caplog.records[0].getMessage()
    assert "private" not in message and "omitted=['updates']" in message


def test_logging_sink_logs_content_only_when_asked_and_only_at_debug(caplog):
    logger = logging.getLogger("test.content")
    with caplog.at_level(logging.DEBUG, logger="test.content"):
        LoggingSink(logger, capture_content=True).emit(ev(arguments={"q": "secret"}))
        LoggingSink(logger, level=logging.INFO, capture_content=True).emit(
            ev(arguments={"q": "hidden"})
        )
    debug, info = (r.getMessage() for r in caplog.records)
    assert "secret" in debug
    assert "hidden" not in info


def test_logging_sink_settings_are_read_only():
    sink = LoggingSink(level=logging.INFO)
    assert sink.level == logging.INFO and sink.logger.name == "agent_adapter.tracing.sinks"
    with pytest.raises(AttributeError):
        sink.level = logging.DEBUG  # type: ignore[misc]


def test_logging_sink_drops_content_keys_above_debug(caplog):
    with caplog.at_level(logging.INFO, logger="agent_adapter.tracing.sinks"):
        LoggingSink(level=logging.INFO).emit(
            ev(
                tool="search",
                arguments={"q": "secret"},
                result="hidden",
                input_messages=["hi"],
                output_messages=["yo"],
            )
        )
    message = caplog.records[0].getMessage()
    assert "search" in message
    for word in ("secret", "hidden", "hi'", "yo'"):
        assert word not in message
    assert "arguments" in message  # key listed as omitted


def test_logging_sink_skips_disabled_levels(caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.tracing.sinks"):
        LoggingSink().emit(ev())
    assert caplog.records == []


def test_logging_sink_never_raises(caplog):
    class Unprintable:
        def __repr__(self):
            raise RuntimeError("no repr")

    logger = logging.getLogger("test.unprintable")
    with caplog.at_level(logging.DEBUG, logger="test.unprintable"):
        LoggingSink(logger).emit(ev(value=Unprintable()))
    assert any("logging sink failed" in r.getMessage() for r in caplog.records)


def test_callback_event_sink_calls_and_swallows_failures(caplog):
    seen = []
    CallbackEventSink(seen.append).emit(ev())
    assert [e.type for e in seen] == ["tool.started"]

    def boom(event):
        raise RuntimeError("down")

    with caplog.at_level(logging.WARNING, logger="agent_adapter.tracing.sinks"):
        CallbackEventSink(boom).emit(ev())
    assert "event callback failed for tool.started" in caplog.text


def test_fanout_sink_sends_to_all_even_when_one_fails(caplog):
    class Broken:
        def emit(self, event):
            raise RuntimeError("down")

    first, last = RecordingSink(), RecordingSink()
    with caplog.at_level(logging.WARNING, logger="agent_adapter.tracing.events"):
        FanoutSink(first, Broken(), last).emit(ev())
    assert first.types() == ["tool.started"] and last.types() == ["tool.started"]
    assert "event sink failed" in caplog.text


def test_fanout_sink_exposes_its_sinks_read_only():
    a, b = RecordingSink(), RecordingSink()
    fanout = FanoutSink(a, b)
    assert fanout.sinks == (a, b)
    with pytest.raises(AttributeError):
        fanout.sinks = ()  # type: ignore[misc]
