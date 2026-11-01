import dataclasses

import pytest

from agent_adapter.core.errors import ToolDefinitionError
from agent_adapter.core.messages import ToolDef
from agent_adapter.core.tool import ToolSpec, build_spec, strict_downgraded


def ping(x: int) -> int:
    """Echo."""
    return x


def test_new_fields_are_appended_with_defaults():
    names = [f.name for f in dataclasses.fields(ToolSpec)]
    assert names[-7:] == [
        "portable",
        "cache_scope",
        "targets",
        "input_schema",
        "output_schema",
        "effects",
        "idempotent",
    ]
    spec = build_spec(ping)
    assert (spec.cache_scope, spec.targets, spec.effects, spec.idempotent) == (
        "run",
        None,
        None,
        None,
    )
    assert spec.input_schema is None and spec.output_schema is None


def test_new_policy_options_are_stored_and_normalised():
    spec = build_spec(
        ping, cache_scope="thread", targets="openai", effects="read_only", idempotent=True
    )
    assert spec.cache_scope == "thread" and spec.targets == frozenset({"openai"})
    assert spec.effects == "read_only" and spec.idempotent is True
    assert build_spec(ping, targets=["mcp", "custom_x"]).targets == frozenset({"mcp", "custom_x"})
    assert build_spec(ping, effects=None, idempotent=None).effects is None
    assert build_spec(ping, targets=None).targets is None
    assert build_spec(ping, tags=[]).tags == frozenset()  # 0.1.0: empty tags stay allowed


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("cache_scope", "session"),
        ("cache_scope", None),
        ("cache_scope", ["run"]),
        ("targets", ""),
        ("targets", []),
        ("targets", ()),
        ("targets", ["openai", ""]),
        ("targets", 3),
        ("effects", "mutating"),
        ("effects", True),
        ("idempotent", "yes"),
        ("idempotent", 1),
    ],
)
def test_bad_new_policy_values_are_rejected(option, value):
    with pytest.raises(ToolDefinitionError, match=f"option {option!r}"):
        build_spec(ping, **{option: value})


def test_input_and_output_schema_override_the_model_schemas():
    raw = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    out = {"type": "object", "properties": {"rows": {"type": "integer"}}}
    spec = dataclasses.replace(build_spec(ping), input_schema=raw, output_schema=out)
    schema = spec.json_schema()
    assert schema == raw and schema is not raw
    schema["properties"]["q"]["type"] = "integer"
    assert spec.json_schema() == raw
    result = spec.result_schema()
    assert result == out
    result["properties"].clear()
    assert spec.result_schema() == out


def test_tool_def_uses_the_neutral_schema():
    spec = build_spec(ping)
    assert spec.tool_def() == ToolDef(
        name="ping", description="Echo.", parameters=spec.json_schema(), strict=True
    )
    assert spec.tool_def(strict=False).strict is False


def test_tool_def_downgrades_only_non_portable_imported_schemas():
    raw = {"type": "object", "properties": {"q": {"type": "string", "pattern": "^a"}}}
    built = build_spec(ping, portable=False)
    assert built.tool_def().strict is True and not strict_downgraded(built)
    imported = dataclasses.replace(built, input_schema=raw)
    assert imported.tool_def().strict is False and strict_downgraded(imported)
    portable = dataclasses.replace(build_spec(ping), input_schema=raw)
    assert portable.tool_def().strict is True
