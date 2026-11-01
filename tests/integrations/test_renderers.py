import sys

import pytest

from agent_adapter import MissingDependency, Registry
from agent_adapter.integrations import anthropic, gemini, openai

SECTION = {"type": "string", "description": "Section number, e.g. 80C."}


def registry() -> Registry:
    reg = Registry()

    @reg.tool
    async def lookup(section: str, year: int = 2025) -> str:
        """Look up a section.

        Args:
            section: Section number, e.g. 80C.
        """
        return section

    return reg


def test_openai_chat_tool_is_strict_function():
    assert openai.chat_tools(registry()) == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a section.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["section", "year"],
                    "properties": {
                        "section": SECTION,
                        "year": {
                            "description": "Default: 2025.",
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                        },
                    },
                },
            },
        }
    ]


def test_openai_non_strict_keeps_defaults_and_optional_fields():
    params = openai.chat_tool(registry().get("lookup"), strict=False)["function"]["parameters"]
    assert params["required"] == ["section"]
    assert params["properties"]["year"] == {"type": "integer", "default": 2025}


def test_openai_responses_tool_is_flat():
    tool = openai.responses_tools(registry())[0]
    assert tool["type"] == "function"
    assert tool["name"] == "lookup"
    assert tool["strict"] is True
    assert "function" not in tool


def test_gemini_function_declaration_uses_json_schema():
    assert gemini.function_declarations(registry()) == [
        {
            "name": "lookup",
            "description": "Look up a section.",
            "parameters_json_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["section"],
                "properties": {
                    "section": SECTION,
                    "year": {"type": "integer", "description": "Default: 2025."},
                },
            },
        }
    ]


def test_gemini_genai_tool_builds_sdk_object():
    pytest.importorskip("google.genai")
    tool = gemini.genai_tool(registry())
    assert tool.function_declarations[0].name == "lookup"


def test_anthropic_tool_uses_input_schema_with_defaults():
    tool = anthropic.anthropic_tools(registry())[0]
    assert tool["name"] == "lookup"
    assert tool["description"] == "Look up a section."
    assert tool["input_schema"]["properties"]["year"] == {"type": "integer", "default": 2025}


def test_gemini_genai_tool_raises_missing_dependency_when_google_genai_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.genai", None)
    with pytest.raises(MissingDependency):
        gemini.genai_tool(registry())
