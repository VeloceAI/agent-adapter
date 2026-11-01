import subprocess
import sys

import agent_adapter.cache as cache
import agent_adapter.cache.providers as providers


def test_cache_package_exports_the_wp05_names():
    assert {"MemoryCache", "RedisCache", "LLMResponseCache", "PromptLayout"} <= set(cache.__all__)
    assert all(hasattr(cache, name) for name in cache.__all__)
    assert set(providers.__all__) == {
        "AnthropicPromptCache",
        "GeminiExplicitCache",
        "GeminiImplicitCache",
        "OpenAIPromptCache",
        "OpenAIPromptCacheV2",
    }


def test_importing_the_cache_package_imports_no_sdk():
    code = (
        "import sys, agent_adapter.cache, agent_adapter.cache.providers; "
        "print(sorted(m for m in ('redis', 'openai', 'anthropic', 'google') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
