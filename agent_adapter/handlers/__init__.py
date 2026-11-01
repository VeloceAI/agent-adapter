"""Handlers: run tool calls from queue messages, serverless events and pipelines."""

from agent_adapter.handlers.consume import Reply, handle_envelope
from agent_adapter.handlers.envelope import MAX_ENVELOPE_BYTES, Envelope, parse_envelope
from agent_adapter.handlers.serverless import Runner, RunSync, tool_handler

__all__ = [
    "MAX_ENVELOPE_BYTES",
    "Envelope",
    "Reply",
    "RunSync",
    "Runner",
    "handle_envelope",
    "parse_envelope",
    "tool_handler",
]
