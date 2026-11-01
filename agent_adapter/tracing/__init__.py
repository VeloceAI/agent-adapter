"""Tracing: the event catalogue, event sinks and the OpenTelemetry sink."""

from agent_adapter.core.ports import EventSink
from agent_adapter.tracing.events import REQUIRED_DATA, EventType, emit, make_event
from agent_adapter.tracing.otel import OTelSink
from agent_adapter.tracing.sinks import (
    CallbackEventSink,
    EventLog,
    FanoutSink,
    LoggingSink,
    SinkHub,
)

__all__ = [
    "REQUIRED_DATA",
    "CallbackEventSink",
    "EventLog",
    "EventSink",
    "EventType",
    "FanoutSink",
    "LoggingSink",
    "OTelSink",
    "SinkHub",
    "emit",
    "make_event",
]
