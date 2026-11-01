"""Every OpenTelemetry attribute, span and metric name the adapter emits (GenAI semconv)."""

# The GenAI conventions are in Development status and still renaming fields: a rename is a
# change to this module only. Names match opentelemetry-semantic-conventions 0.66b.

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

# Operation names (`gen_ai.operation.name` values).
OP_CHAT = "chat"
OP_GENERATE_CONTENT = "generate_content"
OP_EMBEDDINGS = "embeddings"
OP_EXECUTE_TOOL = "execute_tool"
OP_INVOKE_WORKFLOW = "invoke_workflow"
OP_NODE = "node"  # adapter-specific span prefix for flow nodes

# GenAI attributes.
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
GEN_AI_WORKFLOW_NAME = "gen_ai.workflow.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
GEN_AI_TOKEN_TYPE = "gen_ai.token.type"
ERROR_TYPE = "error.type"

# Opt-in content attributes (only with capture_content=True).
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"

# Values.
TOOL_TYPE_FUNCTION = "function"
TOKEN_TYPE_INPUT = "input"
TOKEN_TYPE_OUTPUT = "output"

# Adapter attributes outside the GenAI conventions.
AA_PREFIX = "agent_adapter."  # span-event attributes: AA_PREFIX + event data key
AA_STAGE = "agent_adapter.stage"
AA_COST_USD = "agent_adapter.cost_usd"
AA_CACHE_HIT = "agent_adapter.cache_hit"
AA_CACHE_HITS = "agent_adapter.cache_hits"  # inputs served from the cache
AA_EMBEDDING_TEXTS = "agent_adapter.embedding.texts"  # a count, never the texts
AA_EMBEDDING_BATCHES = "agent_adapter.embedding.batches"
AA_TOOL_KIND = "agent_adapter.tool.kind"
AA_ATTEMPTS = "agent_adapter.attempts"
AA_APPROVAL_ID = "agent_adapter.approval_id"
AA_FLOW_NODE = "agent_adapter.flow.node"
AA_FLOW_VISIT = "agent_adapter.flow.visit"
AA_FLOW_STATUS = "agent_adapter.flow.status"
AA_FLOW_STEPS = "agent_adapter.flow.steps"

# Metrics.
METRIC_TOKEN_USAGE = "gen_ai.client.token.usage"
METRIC_OPERATION_DURATION = "gen_ai.client.operation.duration"

# Adapter provider id -> `gen_ai.provider.name`; others pass through unchanged.
PROVIDER_NAMES: Mapping[str, str] = MappingProxyType(
    {"openai": "openai", "anthropic": "anthropic", "gemini": "gcp.gemini"}
)


def provider_name(provider: str) -> str:
    """The `gen_ai.provider.name` value for an adapter provider id."""
    return PROVIDER_NAMES.get(provider, provider)


def operation_for(provider: str) -> str:
    """The inference operation name: `generate_content` for Gemini, else `chat`."""
    return OP_GENERATE_CONTENT if provider == "gemini" else OP_CHAT
