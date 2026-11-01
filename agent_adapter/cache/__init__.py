"""Caching: backends, keys, envelopes, the tool and LLM caches, and the prompt layout.

Provider prompt-cache plug-ins live in `agent_adapter.cache.providers`.
"""

from agent_adapter.cache.codec import Envelope, decode, encode
from agent_adapter.cache.keys import (
    KEY_VERSION,
    canonical_json,
    digest,
    gcache_key,
    llm_key,
    tool_key,
)
from agent_adapter.cache.layout import PromptLayout
from agent_adapter.cache.llm_cache import LLMResponseCache
from agent_adapter.cache.memory import MemoryCache
from agent_adapter.cache.redis import RedisCache
from agent_adapter.cache.tool_cache import ToolResultCache
from agent_adapter.core.ports import CachePort

__all__ = [
    "KEY_VERSION",
    "CachePort",
    "Envelope",
    "LLMResponseCache",
    "MemoryCache",
    "PromptLayout",
    "RedisCache",
    "ToolResultCache",
    "canonical_json",
    "decode",
    "digest",
    "encode",
    "gcache_key",
    "llm_key",
    "tool_key",
]
