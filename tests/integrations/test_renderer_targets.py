import pytest

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.imported import spec_from_schema
from agent_adapter.core.tool import build_spec
from agent_adapter.integrations import anthropic, gemini, openai
from tests.integrations.sample_tools import lookup
from tests.integrations.sample_tools import targeted_registry as registry


def names(tools, key=None):
    return [key(t) if key else t["name"] for t in tools]


def test_registry_renderers_honour_targets():
    reg = registry()
    assert names(openai.chat_tools(reg), lambda t: t["function"]["name"]) == [
        "lookup",
        "only_openai",
    ]
    assert names(openai.responses_tools(reg)) == ["lookup", "only_openai"]
    assert names(gemini.function_declarations(reg)) == ["lookup", "only_gemini"]
    assert names(anthropic.anthropic_tools(reg)) == ["lookup", "only_anthropic"]


@pytest.mark.parametrize(
    "render",
    [
        openai.chat_tool,
        openai.responses_tool,
        gemini.function_declaration,
        anthropic.anthropic_tool,
    ],
)
def test_single_spec_renderers_reject_excluded_targets(render):
    spec = build_spec(lookup, targets="mcp")
    with pytest.raises(ToolUsageError, match="not exported"):
        render(spec)


NOT_PORTABLE = {"type": "object", "properties": {"q": {"type": "string", "pattern": "^a"}}}


def test_openai_renders_non_portable_imported_tools_non_strict():
    spec = spec_from_schema("remote", "Remote.", NOT_PORTABLE, lambda **kw: kw)
    chat = openai.chat_tool(spec)
    assert chat["function"]["strict"] is False
    assert chat["function"]["parameters"]["properties"]["q"]["pattern"] == "^a"
    assert openai.responses_tool(spec, strict=True)["strict"] is False


def test_openai_keeps_strict_for_portable_imported_and_built_tools():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    assert openai.chat_tool(spec_from_schema("r", "R.", schema, lambda **kw: kw))["function"][
        "strict"
    ]
    assert openai.responses_tool(build_spec(lookup, portable=False))["strict"] is True
