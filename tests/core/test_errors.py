from agent_adapter.core.errors import (
    AdapterError,
    DuplicateTool,
    MissingDependency,
    SchemaNotPortable,
    ToolArgsInvalid,
    ToolDefinitionError,
    ToolNotFound,
    ToolUsageError,
)


def test_errors_share_adapter_error_base():
    for cls in (ToolDefinitionError, DuplicateTool, ToolNotFound):
        assert issubclass(cls, AdapterError)
    assert isinstance(ToolArgsInvalid("t", []), AdapterError)
    assert isinstance(SchemaNotPortable(["x"]), AdapterError)


def test_schema_not_portable_message_names_tool_and_problems():
    err = SchemaNotPortable(["$.a: open map (dict) is not portable"], tool="search")
    assert "tool 'search'" in str(err) and "open map" in str(err)
    assert err.problems == ["$.a: open map (dict) is not portable"]


def test_missing_dependency_subclasses_adapter_error_and_import_error():
    assert issubclass(MissingDependency, AdapterError)
    assert issubclass(MissingDependency, ImportError)


def test_tool_usage_error_is_an_adapter_error_and_a_type_error():
    import agent_adapter

    assert issubclass(ToolUsageError, AdapterError)
    assert issubclass(ToolUsageError, TypeError)
    assert agent_adapter.errors.ToolUsageError is ToolUsageError


def test_args_invalid_normalises_loc_and_drops_input():
    err = ToolArgsInvalid("t", [{"type": "x", "loc": ("a", 0), "msg": "bad", "input": "pii"}])
    assert err.errors == [{"type": "x", "loc": ["a", 0], "msg": "bad"}]
    assert str(err) == "invalid arguments for tool 't': a.0: bad"


def test_args_invalid_root_loc_is_dollar():
    assert str(ToolArgsInvalid("t", [{"type": "x", "loc": (), "msg": "bad"}])).endswith("$: bad")
