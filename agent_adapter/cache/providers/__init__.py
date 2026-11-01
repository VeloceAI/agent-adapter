"""Provider prompt-cache plug-ins; each edits SDK kwargs only where the model's caps allow.

Without `caps=`, the capability flags come from `Prices.default().caps`.
"""

from agent_adapter.cache.providers.anthropic import AnthropicPromptCache
from agent_adapter.cache.providers.gemini import GeminiExplicitCache, GeminiImplicitCache
from agent_adapter.cache.providers.openai import OpenAIPromptCache, OpenAIPromptCacheV2

__all__ = [
    "AnthropicPromptCache",
    "GeminiExplicitCache",
    "GeminiImplicitCache",
    "OpenAIPromptCache",
    "OpenAIPromptCacheV2",
]
