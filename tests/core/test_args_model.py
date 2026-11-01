from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolArgsInvalid, ToolDefinitionError
from agent_adapter.core.tool import build_spec


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="What to look for.")
    top_k: int = 5


def legacy_search(query: str, top_k: Any, ctx: ToolContext, debug: bool = False):
    """Search the legacy index."""
    return [query] * top_k, ctx.run_id, debug


async def test_args_model_replaces_the_signature_schema():
    spec = build_spec(legacy_search, args_model=SearchArgs, hide="debug")
    assert spec.args_model is SearchArgs
    schema = spec.json_schema()
    assert schema["properties"]["query"]["description"] == "What to look for."
    assert schema["properties"]["top_k"]["default"] == 5
    assert spec.context_param == "ctx"
    result = await spec.invoke({"query": "80C", "top_k": 2}, ToolContext(run_id="r"))
    assert result == (["80C", "80C"], "r", False)


async def test_args_model_works_with_rename_and_fixed():
    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        q: str

    def find(text: str, limit: int) -> list[str]:
        """Find."""
        return [text] * limit

    spec = build_spec(find, args_model=Args, rename={"text": "q"}, fixed={"limit": 2})
    assert await spec.invoke({"q": "x"}) == ["x", "x"]
    assert spec.llm_to_python == {"q": "text"}
    with pytest.raises(ToolArgsInvalid):
        await spec.invoke({"q": "x", "limit": 3})


async def test_strict_nulls_still_apply():
    spec = build_spec(legacy_search, args_model=SearchArgs, hide="debug")
    result = await spec.invoke({"query": "a", "top_k": None}, ToolContext())
    assert result[0] == ["a"] * 5


def test_field_names_must_match_the_llm_facing_parameters():
    class Wrong(BaseModel):
        model_config = ConfigDict(extra="forbid")
        query: str
        limit: int

    with pytest.raises(ToolDefinitionError, match=r"missing \['top_k'\], extra \['limit'\]"):
        build_spec(legacy_search, args_model=Wrong, hide="debug")
    with pytest.raises(ToolDefinitionError, match=r"missing \['debug'\]"):
        build_spec(legacy_search, args_model=SearchArgs)


def test_args_model_must_forbid_extra_fields():
    class Loose(BaseModel):
        query: str
        top_k: int

    with pytest.raises(ToolDefinitionError, match="extra='forbid'"):
        build_spec(legacy_search, args_model=Loose, hide="debug")


def test_args_model_must_not_use_aliases():
    class Aliased(BaseModel):
        model_config = ConfigDict(extra="forbid")
        query: str = Field(alias="q")
        top_k: int

    with pytest.raises(ToolDefinitionError, match="sets an alias"):
        build_spec(legacy_search, args_model=Aliased, hide="debug")


@pytest.mark.parametrize("value", [dict, SearchArgs(query="x"), "SearchArgs"])
def test_args_model_must_be_a_model_class(value):
    with pytest.raises(ToolDefinitionError, match="BaseModel subclass"):
        build_spec(legacy_search, args_model=value, hide="debug")


def test_names_are_still_checked():
    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def bad(_private: int):
        """Bad."""

    with pytest.raises(ToolDefinitionError, match="must not start with '_'"):
        build_spec(bad, args_model=Args)


def test_annotations_stay_required_with_an_args_model():
    def untyped(query, top_k):  # type: ignore[no-untyped-def]
        """Untyped."""

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        query: str
        top_k: int

    with pytest.raises(ToolDefinitionError, match="needs a type annotation"):
        build_spec(untyped, args_model=Args)


def test_model_types_drive_validation():
    spec = build_spec(legacy_search, args_model=SearchArgs, hide="debug")
    with pytest.raises(ToolArgsInvalid):
        spec.validate_args({"query": "x", "top_k": "many"})
