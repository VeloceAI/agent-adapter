import pytest

from agent_adapter.tracing import semconv


def test_contract_names():
    assert semconv.GEN_AI_OPERATION_NAME == "gen_ai.operation.name"
    assert semconv.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS == "gen_ai.usage.cache_read.input_tokens"
    assert (
        semconv.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
        == "gen_ai.usage.cache_creation.input_tokens"
    )
    assert semconv.ERROR_TYPE == "error.type"
    assert semconv.METRIC_TOKEN_USAGE == "gen_ai.client.token.usage"
    assert semconv.METRIC_OPERATION_DURATION == "gen_ai.client.operation.duration"


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("gemini", "gcp.gemini"),
        ("voyage", "voyage"),
    ],
)
def test_provider_names(provider, expected):
    assert semconv.provider_name(provider) == expected


def test_operation_for_provider():
    assert semconv.operation_for("gemini") == "generate_content"
    assert semconv.operation_for("openai") == "chat"
    assert semconv.operation_for("callable") == "chat"


GEN_AI = [name for name in dir(semconv) if name.startswith("GEN_AI_")]


@pytest.mark.parametrize("name", GEN_AI)
def test_names_match_the_installed_semantic_conventions(name):
    ga = pytest.importorskip("opentelemetry.semconv._incubating.attributes.gen_ai_attributes")
    assert getattr(ga, name) == getattr(semconv, name)


def test_metric_and_error_names_match_the_installed_semantic_conventions():
    gm = pytest.importorskip("opentelemetry.semconv._incubating.metrics.gen_ai_metrics")
    errors = pytest.importorskip("opentelemetry.semconv.attributes.error_attributes")
    assert gm.GEN_AI_CLIENT_TOKEN_USAGE == semconv.METRIC_TOKEN_USAGE
    assert gm.GEN_AI_CLIENT_OPERATION_DURATION == semconv.METRIC_OPERATION_DURATION
    assert errors.ERROR_TYPE == semconv.ERROR_TYPE


def test_operation_names_match_the_installed_semantic_conventions():
    ga = pytest.importorskip("opentelemetry.semconv._incubating.attributes.gen_ai_attributes")
    values = ga.GenAiOperationNameValues
    assert semconv.OP_EMBEDDINGS == values.EMBEDDINGS.value == "embeddings"
    assert values.CHAT.value == semconv.OP_CHAT
    assert values.GENERATE_CONTENT.value == semconv.OP_GENERATE_CONTENT
    assert values.EXECUTE_TOOL.value == semconv.OP_EXECUTE_TOOL
