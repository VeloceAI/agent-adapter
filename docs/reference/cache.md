# Cache

Tool-result and model-response caches, keys and provider prompt caches. Concepts: [Cache](../concepts/cache.md).

## `agent_adapter.cache`

::: agent_adapter.cache
    options:
      members: [CachePort, Envelope, KEY_VERSION, LLMResponseCache, PromptLayout, RedisCache, ToolResultCache, canonical_json, decode, digest, encode, gcache_key, llm_key, tool_key]

Also exported here, documented under the canonical path: [`MemoryCache`](top-level.md#agent_adapter.MemoryCache).

## `agent_adapter.cache.providers`

::: agent_adapter.cache.providers
    options:
      members: [AnthropicPromptCache, GeminiExplicitCache, GeminiImplicitCache, OpenAIPromptCache, OpenAIPromptCacheV2]
