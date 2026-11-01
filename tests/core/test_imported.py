import logging

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import SchemaNotPortable, ToolArgsInvalid, ToolDefinitionError
from agent_adapter.core.imported import spec_from_schema
from agent_adapter.core.registry import Registry

SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search text."},
        "max-results": {"type": "integer", "default": 5},
        "region": {"type": "string"},
    },
    "required": ["query"],
}
OUTPUT = {"type": "object", "properties": {"hits": {"type": "array", "items": {"type": "string"}}}}


async def test_func_gets_set_values_by_property_name_without_rejected_nulls():
    received = []

    async def remote(**kwargs):
        received.append(kwargs)
        return "ok"

    spec = spec_from_schema("search_docs", "Search docs.", SCHEMA, remote)
    assert spec.args_model.__name__ == "SearchDocsArgs"
    assert spec.llm_to_python == {"max_results": "max-results"}
    assert await spec.invoke({"query": "x"}) == "ok"
    assert await spec.invoke({"query": "y", "max-results": 2, "region": None}) == "ok"
    assert received == [{"query": "x"}, {"query": "y", "max-results": 2}]


async def test_sync_funcs_and_context_param():
    def remote(query, *, ctx):
        return query, ctx.run_id

    schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    spec = spec_from_schema("s", "S.", schema, remote, context_param="ctx")
    assert spec.context_param == "ctx"
    assert await spec.invoke({"query": "q"}, ToolContext(run_id="r")) == ("q", "r")


async def test_required_nullable_property_is_passed_as_null():
    async def remote(**kwargs):
        return kwargs

    schema = {
        "type": "object",
        "properties": {"region": {"type": ["string", "null"]}},
        "required": ["region"],
    }
    spec = spec_from_schema("s", "S.", schema, remote)
    assert await spec.invoke({"region": None}) == {"region": None}


async def test_open_schemas_forward_extra_properties():
    async def remote(**kwargs):
        return kwargs

    schema = {"type": "object", "properties": {}, "additionalProperties": True}
    spec = spec_from_schema("s", "S.", schema, remote)
    assert await spec.invoke({"anything": 1}) == {"anything": 1}


async def test_arguments_are_validated_loosely():
    spec = spec_from_schema("s", "S.", SCHEMA, lambda **kw: kw)
    with pytest.raises(ToolArgsInvalid):
        await spec.invoke({})
    with pytest.raises(ToolArgsInvalid):
        await spec.invoke({"query": "x", "unknown": 1})


def test_schemas_are_stored_as_copies_and_returned_as_is():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    output = dict(OUTPUT)
    spec = spec_from_schema("s", "  S.  ", schema, lambda **kw: kw, output_schema=output)
    schema["properties"]["q"]["type"] = "integer"
    output["type"] = "string"
    assert spec.json_schema() == {"type": "object", "properties": {"q": {"type": "string"}}}
    assert spec.result_schema() == OUTPUT
    assert spec.description == "S."
    assert spec_from_schema("s", "S.", schema, lambda **kw: kw).result_schema() is None


def test_policy_options_apply():
    spec = spec_from_schema(
        "s", "S.", SCHEMA, lambda **kw: kw, kind="mcp", tags="mcp", effects="read_only"
    )
    assert (spec.kind, spec.tags, spec.effects) == ("mcp", frozenset({"mcp"}), "read_only")
    with pytest.raises(ToolDefinitionError, match="option 'effects'"):
        spec_from_schema("s", "S.", SCHEMA, lambda **kw: kw, effects="bad")


NOT_PORTABLE = {"type": "object", "properties": {"q": {"type": "string", "pattern": "^a"}}}


def test_portability_is_detected_when_not_given(caplog):
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.core.imported"):
        spec = spec_from_schema("s", "S.", NOT_PORTABLE, lambda **kw: kw)
    assert spec.portable is False
    assert "regex pattern is not portable" in caplog.text
    assert spec_from_schema("s", "S.", SCHEMA, lambda **kw: kw).portable is True
    Registry().add(spec)  # non-portable imported tools register; renderers relax strictness


def test_portable_true_keeps_the_registration_check_strict():
    spec = spec_from_schema("s", "S.", NOT_PORTABLE, lambda **kw: kw, portable=True)
    with pytest.raises(SchemaNotPortable):
        Registry().add(spec)
    relaxed = spec_from_schema("s", "S.", NOT_PORTABLE, lambda **kw: kw, portable=False)
    assert Registry().add(relaxed) is relaxed


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("bad name", "S.", SCHEMA), "invalid tool name"),
        ((3, "S.", SCHEMA), "invalid tool name"),
        (("s", "  ", SCHEMA), "non-empty description"),
        (("s", None, SCHEMA), "non-empty description"),
        (("s", "S.", {"type": "string"}), "root must be an object"),
        (("s", "S.", ["object"]), "root must be an object"),
    ],
)
def test_bad_inputs_are_rejected(args, message):
    with pytest.raises(ToolDefinitionError, match=message):
        spec_from_schema(*args, lambda **kw: kw)


def test_bad_func_portable_and_context_param_are_rejected():
    with pytest.raises(ToolDefinitionError, match="func must be callable"):
        spec_from_schema("s", "S.", SCHEMA, "not callable")
    with pytest.raises(ToolDefinitionError, match="portable must be"):
        spec_from_schema("s", "S.", SCHEMA, lambda **kw: kw, portable="yes")
    with pytest.raises(ToolDefinitionError, match="identifier"):
        spec_from_schema("s", "S.", SCHEMA, lambda **kw: kw, context_param="a-b")
    with pytest.raises(ToolDefinitionError, match="also a schema property"):
        spec_from_schema("s", "S.", SCHEMA, lambda **kw: kw, context_param="query")


async def test_nulls_are_kept_where_the_property_schema_allows_them():
    async def remote(**kwargs):
        return kwargs

    schema = {
        "type": "object",
        "properties": {
            "a": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "b": {"type": "string", "nullable": True},
            "c": {},
            "d": {"type": "string"},
        },
    }
    spec = spec_from_schema("s", "S.", schema, remote)
    assert await spec.invoke({"a": None, "b": None, "c": None, "d": None}) == {
        "a": None,
        "b": None,
        "c": None,
    }


async def test_field_names_validate_too():
    async def remote(**kwargs):
        return kwargs

    spec = spec_from_schema("s", "S.", SCHEMA, remote)
    assert await spec.invoke({"query": "x", "max_results": 3}) == {"query": "x", "max-results": 3}


def counting_builds(monkeypatch):
    """Count the argument models `spec_from_schema` tools build."""
    from agent_adapter.core import imported

    built = []
    real = imported.model_from_json_schema

    def build(name, schema):
        built.append(name)
        return real(name, schema)

    monkeypatch.setattr(imported, "model_from_json_schema", build)
    return built


async def test_the_argument_model_is_built_on_first_use_then_cached(monkeypatch):
    from agent_adapter.schema.models import model_from_json_schema

    built = counting_builds(monkeypatch)
    spec = spec_from_schema("search_docs", "Search docs.", SCHEMA, lambda **kw: kw)
    Registry().add(spec)
    assert built == [] and spec.json_schema() == SCHEMA
    assert spec.llm_to_python == {"max_results": "max-results"}
    eager = model_from_json_schema("SearchDocsArgs", SCHEMA)
    validated = spec.validate_args({"query": "q", "max-results": 2})
    assert built == ["SearchDocsArgs"]
    assert (
        validated.model_dump()
        == eager.model_validate({"query": "q", "max-results": 2}).model_dump()
    )
    assert spec.args_model is spec.args_model and built == ["SearchDocsArgs"]
    assert set(spec.args_model.model_fields) == set(eager.model_fields)
    with pytest.raises(ToolArgsInvalid):
        spec.validate_args({})


def test_concurrent_first_uses_build_one_model(monkeypatch):
    import threading

    built = counting_builds(monkeypatch)
    spec = spec_from_schema("search_docs", "Search docs.", SCHEMA, lambda **kw: kw)
    barrier = threading.Barrier(8)
    seen = []

    def use():
        barrier.wait()
        seen.append(spec.args_model)

    workers = [threading.Thread(target=use) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert len(built) == 1 and len({id(model) for model in seen}) == 1
