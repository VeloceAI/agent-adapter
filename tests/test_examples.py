import json
import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _run(name: str, capsys) -> str:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out


def test_quickstart_runs(capsys):
    out = _run("quickstart.py", capsys)
    assert "Article returns (en)" in out
    assert '"name": "shipping_rate"' in out
    assert "18.0" in out


async def test_quickstart_executor_denies_a_confirm_tool_the_model_calls():
    from agent_adapter import Executor, ToolCall

    ns = runpy.run_path(str(EXAMPLES / "quickstart.py"))
    assert isinstance(ns["executor"], Executor)
    ran: list[str] = []

    def wipe(target: str) -> None:
        """Delete a target."""
        ran.append(target)

    ns["registry"].wrap(wipe, confirm=True, effects="destructive")
    result = await ns["executor"].run_call(ToolCall(name="wipe", arguments={"target": "db"}))
    assert result.error is not None and result.error.kind == "approval-denied"
    assert ran == []


def test_providers_renders_each_shape_and_runs_each_call(capsys):
    out = _run("providers.py", capsys)
    assert '"function": {' in out
    assert '"parameters_json_schema": {' in out
    assert '"input_schema": {' in out
    assert 'Default: \\"c\\".' in out
    assert out.count("Paris: 31 degrees C") == 2
    assert "Paris: 31 degrees F" in out


def test_wrap_existing_shapes_and_calls_legacy_code(capsys):
    lines = _run("wrap_existing.py", capsys).splitlines()
    assert lines[:6] == [
        "['articles']",
        "{'articles': 'article_list'}",
        "['Article returns (en)']",
        "18.0",
        "7.9",
        "9.2",
    ]
    assert json.loads(lines[6]) == ["get_articles", "shipping_rate", "usd_to_gbp", "usd_to_eur"]


def test_strict_mode_requires_every_field_and_applies_defaults(capsys):
    out = _run("strict_mode.py", capsys)
    schema_text, count, first = out.rsplit("\n", 3)[:3]
    schema = json.loads(schema_text)
    assert schema["required"] == ["query", "top_k", "region"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["top_k"]["description"] == "How many hits. Default: 8."
    assert count == "8"
    assert first == "returns #0 (None)"


def test_errors_reports_each_error_type(capsys):
    lines = _run("errors.py", capsys).splitlines()
    reply = json.loads(lines[0])
    assert reply["tool"] == "add"
    assert [e["loc"] for e in reply["errors"]] == [["b"], ["c"]]
    assert "two" not in lines[0]
    assert lines[1] == "tag_counts ['$.counts: open map (dict) is not portable']"
    assert lines[2] == "tool 'no_docstring' needs a description or a docstring"
    assert lines[3].startswith("no tool named 'subtract'")


def test_registry_views_filters_and_renders_catalogue(capsys):
    out = _run("registry_views.py", capsys)
    lines = out.splitlines()
    assert lines[0] == "['search']"
    assert lines[1] == "[]"
    assert "When to use: First step for any query." in out
    assert "- `top_k` (optional): How many hits." in out
    assert lines[-1] == "delete_note False"


def test_imported_tools_send_set_values_and_honour_targets(capsys):
    lines = _run("imported_tools.py", capsys).splitlines()
    assert lines[0] == "{'query': 'returns'}"
    assert lines[1] == "{'query': 'returns', 'max-results': 2}"
    assert lines[2] == "['remote_search']"
    assert lines[3] == "['remote_search', 'legacy_lookup']"
    assert lines[4] == "['remote_search']"
    assert json.loads(lines[5]) == ["query"]


def test_runtime_runs_async_tools_from_sync_code(capsys):
    lines = _run("runtime.py", capsys).splitlines()
    assert lines == ["GET /returns", "GET /returns", "[True, True]", "run_"]


def test_context_emits_events_and_describes_errors(capsys):
    lines = _run("context.py", capsys).splitlines()
    assert lines[:3] == ["custom.progress {'pct': 50}", "42", "run_1 call_2 None"]
    info = json.loads(lines[4])
    assert info["kind"] == "tool-error" and info["details"] == {"source": "broken"}
    assert lines[5] == "KeyError"


def test_cache_example_shows_backends_layout_and_plugins(capsys):
    lines = _run("cache.py", capsys).splitlines()
    assert lines[0] == "b'hello' None"
    assert lines[1] == "Paris True 0.0"
    assert lines[2].startswith("system True 'You answer customer-support questions")
    assert lines[3] == r"user False 'Customer plan: premium.\n\nWhat is the returns policy?'"
    assert lines[4] == "['model', 'prompt_cache_key', 'prompt_cache_retention']"
    assert lines[5] == "{'type': 'ephemeral', 'ttl': '1h'}"
    assert lines[6] == "True"


def test_cache_example_builds_a_lazy_redis_cache():
    pytest.importorskip("redis")
    from agent_adapter.cache import RedisCache

    namespace = runpy.run_path(str(EXAMPLES / "cache.py"))
    cache = namespace["shared_cache"]()
    assert isinstance(cache, RedisCache) and cache.prefix == "myapp:"


def test_models_wraps_host_function_falls_back_and_normalises_usage(capsys):
    lines = _run("models.py", capsys).splitlines()
    assert lines[0] == "host answer to 'Return window?'"
    assert lines[1] == "1000 4000 7000"  # uncached = total - cached reads - cache writes
    if len(lines) > 2:  # every provider SDK installed
        assert lines[2] == "['OpenAIModel', 'OpenAIModel', 'GeminiModel', 'AnthropicModel']"
