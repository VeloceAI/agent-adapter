"""Model ports: provider adapters, a host-function port, fallback, instrumentation, usage.

The provider ports import their SDK only when used (`pip install "agent-adapter[openai]"`,
`[gemini]`, `[anthropic]`).
"""

from agent_adapter.core.messages import OutputSchema
from agent_adapter.core.ports import CapabilityLookup, LLMPort, PromptCachePlugin
from agent_adapter.core.types import ModelCaps
from agent_adapter.models.anthropic import AnthropicModel
from agent_adapter.models.callable import CallableModel
from agent_adapter.models.fallback import DEFAULT_FALL_ON, FallbackModel
from agent_adapter.models.gemini import GeminiModel
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.models.openai import OpenAIModel
from agent_adapter.models.usage import (
    normalise_usage,
    usage_anthropic,
    usage_gemini,
    usage_openai_chat,
    usage_openai_responses,
    usage_voyage,
)

__all__ = [
    "DEFAULT_FALL_ON",
    "AnthropicModel",
    "CallableModel",
    "CapabilityLookup",
    "FallbackModel",
    "GeminiModel",
    "InstrumentedModel",
    "LLMPort",
    "ModelCaps",
    "OpenAIModel",
    "OutputSchema",
    "PromptCachePlugin",
    "normalise_usage",
    "usage_anthropic",
    "usage_gemini",
    "usage_openai_chat",
    "usage_openai_responses",
    "usage_voyage",
]
