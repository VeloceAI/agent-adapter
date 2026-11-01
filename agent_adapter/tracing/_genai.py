"""Pure mapping from adapter events to GenAI span names and attributes (no OTel import)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.core.types import Event
from agent_adapter.tracing import semconv as sc

# Scalar, non-content keys copied onto span events; never reasons, names or free text.
NOTE_KEYS = frozenset({"attempt", "delay_s", "error_kind", "approval_id", "approved"})
_FAILED_FLOW = frozenset({"failed", "error", "cancelled"})
_PROMPT_KEYS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens")

Attrs = dict[str, Any]
Built = tuple[Attrs, str | None]  # closing attributes and the error kind, if any


@dataclass(frozen=True)
class SpanStart:
    """What a starting event opens: span name, attributes, metric attributes, client kind."""

    name: str
    attrs: Attrs
    metric_attrs: Attrs = field(default_factory=dict)
    client: bool = False


def _usage_counts(value: Any) -> Mapping[str, Any]:
    """Usage as a mapping, whether the event carries a `Usage` or its dict dump."""
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else {}


def _count(usage: Mapping[str, Any], key: str) -> int:
    return int(usage.get(key) or 0)


def _content(attrs: Attrs, name: str, data: Mapping[str, Any], key: str, capture: bool) -> None:
    value = data.get(key)
    if capture and value is not None:
        text = (
            value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
        )
        attrs[name] = text


def base(event: Event) -> Attrs:
    """Attributes every span carries: the run as the conversation id."""
    return {sc.GEN_AI_CONVERSATION_ID: event.run_id} if event.run_id else {}


def llm_start(event: Event, capture: bool) -> SpanStart:
    """`chat {model}` / `generate_content {model}` span for an llm event."""
    data = event.data
    provider = str(data["provider"])
    model = str(data["model"])
    op = sc.operation_for(provider)
    metric_attrs: Attrs = {
        sc.GEN_AI_OPERATION_NAME: op,
        sc.GEN_AI_PROVIDER_NAME: sc.provider_name(provider),
        sc.GEN_AI_REQUEST_MODEL: model,
    }
    attrs = {
        **base(event),
        **metric_attrs,
        sc.GEN_AI_REQUEST_STREAM: bool(data.get("stream", False)),
        sc.AA_STAGE: str(data.get("stage", "")),
    }
    _content(attrs, sc.GEN_AI_INPUT_MESSAGES, data, "input_messages", capture)
    return SpanStart(f"{op} {model}", attrs, metric_attrs, client=provider != "callable")


def llm_finish(event: Event, capture: bool) -> Built:
    """Response, usage and cost attributes of `llm.finished`."""
    data = event.data
    float(data["latency_ms"])  # a bad latency fails here, before the span closes
    usage = _usage_counts(data.get("usage"))
    attrs: Attrs = {
        sc.GEN_AI_RESPONSE_MODEL: str(data.get("response_model") or data["model"]),
        sc.GEN_AI_RESPONSE_FINISH_REASONS: [str(data["finish_reason"])],
        # OTel counts cache reads and writes inside the input tokens.
        sc.GEN_AI_USAGE_INPUT_TOKENS: sum(_count(usage, key) for key in _PROMPT_KEYS),
        sc.GEN_AI_USAGE_OUTPUT_TOKENS: _count(usage, "output_tokens"),
        sc.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: _count(usage, "cached_input_tokens"),
        sc.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS: _count(usage, "cache_write_input_tokens"),
        sc.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS: _count(usage, "reasoning_tokens"),
    }
    if data.get("response_id"):
        attrs[sc.GEN_AI_RESPONSE_ID] = str(data["response_id"])
    if data.get("cost_usd") is not None:
        attrs[sc.AA_COST_USD] = float(data["cost_usd"])
    _content(attrs, sc.GEN_AI_OUTPUT_MESSAGES, data, "output_messages", capture)
    return attrs, None


def embedding_start(event: Event) -> SpanStart:
    """`embeddings {model}` span; carries the text count, never the texts."""
    data = event.data
    provider = str(data["provider"])
    model = str(data["model"])
    metric_attrs: Attrs = {
        sc.GEN_AI_OPERATION_NAME: sc.OP_EMBEDDINGS,
        sc.GEN_AI_PROVIDER_NAME: sc.provider_name(provider),
        sc.GEN_AI_REQUEST_MODEL: model,
    }
    attrs = {
        **base(event),
        **metric_attrs,
        sc.AA_STAGE: str(data.get("stage", "")),
        sc.AA_EMBEDDING_TEXTS: int(data["texts"]),
    }
    name = f"{sc.OP_EMBEDDINGS} {model}"
    return SpanStart(name, attrs, metric_attrs, client=provider != "callable")


def embedding_finish(event: Event) -> Built:
    """Response model, input tokens, cache hits, batches and cost of `embedding.finished`."""
    data = event.data
    float(data["latency_ms"])  # a bad latency fails here, before the span closes
    usage = _usage_counts(data.get("usage"))
    batches = int(data["batches"])
    attrs: Attrs = {
        sc.GEN_AI_RESPONSE_MODEL: str(data.get("response_model") or data["model"]),
        sc.GEN_AI_USAGE_INPUT_TOKENS: sum(_count(usage, key) for key in _PROMPT_KEYS),
        sc.AA_CACHE_HITS: int(data["cache_hits"]),
        sc.AA_CACHE_HIT: batches == 0,
        sc.AA_EMBEDDING_BATCHES: batches,
    }
    if data.get("cost_usd") is not None:
        attrs[sc.AA_COST_USD] = float(data["cost_usd"])
    return attrs, None


def tool_start(event: Event, capture: bool) -> SpanStart:
    """`execute_tool {tool}` span."""
    data = event.data
    tool = str(data["tool"])
    attrs = {
        **base(event),
        sc.GEN_AI_OPERATION_NAME: sc.OP_EXECUTE_TOOL,
        sc.GEN_AI_TOOL_NAME: tool,
        sc.GEN_AI_TOOL_TYPE: sc.TOOL_TYPE_FUNCTION,
    }
    if event.call_id:
        attrs[sc.GEN_AI_TOOL_CALL_ID] = event.call_id
    if data.get("kind"):
        attrs[sc.AA_TOOL_KIND] = str(data["kind"])
    _content(attrs, sc.GEN_AI_TOOL_CALL_ARGUMENTS, data, "arguments", capture)
    return SpanStart(f"{sc.OP_EXECUTE_TOOL} {tool}", attrs)


def tool_finish(event: Event, capture: bool) -> Built:
    """Attempts, cache hit and (opt-in) result of `tool.finished`."""
    data = event.data
    attrs: Attrs = {sc.AA_ATTEMPTS: int(data["attempts"]), sc.AA_CACHE_HIT: bool(data["cached"])}
    _content(attrs, sc.GEN_AI_TOOL_CALL_RESULT, data, "result", capture)
    return attrs, None


def tool_failed(event: Event) -> Built:
    """Attempts and error kind of `tool.failed`."""
    return {sc.AA_ATTEMPTS: int(event.data["attempts"])}, str(event.data["error_kind"])


def approval(event: Event) -> Built:
    """The approval id a pending tool or paused node ends with."""
    return {sc.AA_APPROVAL_ID: str(event.data["approval_id"])}, None


COMPRESS_CONTEXT = "compress_context"  # adapter-specific span name for context compression
_CONTEXT_COUNTS = (
    "tokens_before",
    "tokens_after",
    "target_tokens",
    "messages_before",
    "messages_after",
)


def context_start(event: Event) -> SpanStart:
    """`compress_context {model}` span: the model whose request was compressed, and why."""
    data = event.data
    model = str(data["model"])
    attrs = {
        **base(event),
        sc.GEN_AI_PROVIDER_NAME: sc.provider_name(str(data["provider"])),
        sc.GEN_AI_REQUEST_MODEL: model,
        f"{sc.AA_PREFIX}context.trigger": str(data["trigger"]),
    }
    return SpanStart(f"{COMPRESS_CONTEXT} {model}", attrs)


def context_finish(event: Event) -> Built:
    """Token and message counts, step names and the summary cost; never message text."""
    data = event.data
    attrs: Attrs = {f"{sc.AA_PREFIX}context.{key}": int(data[key]) for key in _CONTEXT_COUNTS}
    attrs[f"{sc.AA_PREFIX}context.steps"] = ",".join(str(step) for step in data["steps"])
    attrs[sc.AA_COST_USD] = float(data["summary_cost_usd"])
    return attrs, None


def failed(event: Event) -> Built:
    """No attributes, just the event's error kind."""
    return {}, str(event.data["error_kind"])


def note(event: Event) -> Attrs:
    """Span-event attributes: only `NOTE_KEYS` scalars."""
    return {
        f"{sc.AA_PREFIX}{key}": value
        for key, value in event.data.items()
        if key in NOTE_KEYS and isinstance(value, str | bool | int | float)
    }


def flow_start(event: Event) -> SpanStart:
    """`invoke_workflow {flow}` span."""
    flow = str(event.data["flow"])
    attrs = {
        **base(event),
        sc.GEN_AI_OPERATION_NAME: sc.OP_INVOKE_WORKFLOW,
        sc.GEN_AI_WORKFLOW_NAME: flow,
    }
    return SpanStart(f"{sc.OP_INVOKE_WORKFLOW} {flow}", attrs)


def flow_finish(event: Event) -> Built:
    """Status and steps; `failed` / `error` statuses are errors."""
    data = event.data
    status = str(data["status"])
    attrs: Attrs = {sc.AA_FLOW_STATUS: status, sc.AA_FLOW_STEPS: int(data["steps"])}
    error = str(data.get("error_kind") or status) if status in _FAILED_FLOW else None
    return attrs, error


def node_start(event: Event) -> SpanStart:
    """`node {flow}.{node}` span."""
    data = event.data
    attrs = {
        **base(event),
        sc.GEN_AI_WORKFLOW_NAME: str(data["flow"]),
        sc.AA_FLOW_NODE: str(data["node"]),
    }
    if "visit" in data:
        attrs[sc.AA_FLOW_VISIT] = int(data["visit"])
    return SpanStart(f"{sc.OP_NODE} {data['flow']}.{data['node']}", attrs)
