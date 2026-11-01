from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolDefinitionError
from agent_adapter.core.tool import build_spec


class Filter(BaseModel):
    court: str = "sc"


async def search(
    query: str, top_k: int = 8, flt: Filter | None = None, ctx: ToolContext | None = None
) -> list[str]:
    """Search articles.

    Args:
        query: The search text.
        top_k: How many hits.
    """
    return []


class Service:
    def lookup(self, section: str) -> str:
        """Look up a section."""
        return section


def test_build_spec_uses_name_docstring_and_signature():
    spec = build_spec(search)
    schema = spec.json_schema()
    assert spec.name == "search"
    assert spec.description == "Search articles."
    assert schema["required"] == ["query"]
    assert set(schema["properties"]) == {"query", "top_k", "flt"}
    assert schema["properties"]["query"]["description"] == "The search text."
    assert schema["properties"]["top_k"]["default"] == 8
    assert spec.context_param == "ctx"


def test_build_spec_explicit_name_and_description_win():
    spec = build_spec(search, name="find", description="Find things.")
    assert (spec.name, spec.description) == ("find", "Find things.")


def test_build_spec_without_description_raises():
    async def nodoc(x: int) -> int:
        return x

    with pytest.raises(ToolDefinitionError, match="description"):
        build_spec(nodoc)


def test_build_spec_rejects_var_args():
    def many(*items: str) -> None:
        """Doc."""

    with pytest.raises(ToolDefinitionError, match=r"\*args"):
        build_spec(many)


def test_build_spec_rejects_missing_annotation():
    def untyped(x) -> None:
        """Doc."""

    with pytest.raises(ToolDefinitionError, match="annotation"):
        build_spec(untyped)


def test_build_spec_rejects_invalid_name():
    with pytest.raises(ToolDefinitionError, match="invalid tool name"):
        build_spec(search, name="bad-name!")


def test_build_spec_fixed_and_hidden_params_leave_schema():
    spec = build_spec(search, fixed={"top_k": 3}, hide=["flt"])
    assert set(spec.json_schema()["properties"]) == {"query"}
    assert spec.fixed == {"top_k": 3}


def test_build_spec_hidden_param_without_default_raises():
    with pytest.raises(ToolDefinitionError, match="needs a default"):
        build_spec(search, hide=["query"])


def test_build_spec_unknown_param_in_options_raises():
    with pytest.raises(ToolDefinitionError, match="unknown parameters"):
        build_spec(search, fixed={"nope": 1})


def test_build_spec_rename_changes_llm_facing_name():
    spec = build_spec(search, rename={"query": "text"})
    assert "text" in spec.json_schema()["properties"]
    assert spec.llm_to_python == {"text": "query"}


def test_build_spec_stores_policy_options():
    spec = build_spec(search, kind="retrieval", tags=["dt"], parallel_safe=True, timeout=5.0)
    assert (spec.kind, spec.tags, spec.parallel_safe, spec.timeout) == (
        "retrieval",
        frozenset({"dt"}),
        True,
        5.0,
    )


def test_build_spec_unknown_policy_option_raises():
    with pytest.raises(ToolDefinitionError, match="unknown option"):
        build_spec(search, colour="red")


def test_build_spec_bound_method_drops_self():
    spec = build_spec(Service().lookup)
    assert spec.name == "lookup"
    assert set(spec.json_schema()["properties"]) == {"section"}
