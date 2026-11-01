import pytest

from agent_adapter.core.errors import DuplicateTool, SchemaNotPortable, ToolUsageError
from agent_adapter.core.registry import Registry, check_target, exported_to
from agent_adapter.core.tool import build_spec


def a(x: int) -> int:
    """A."""
    return x


def b(x: int) -> int:
    """B."""
    return x


def counts(counts: dict[str, int]) -> int:
    """Open map: not portable."""
    return len(counts)


def test_add_many_registers_all_in_order():
    reg = Registry()
    specs = reg.add_many(build_spec(f) for f in (a, b))
    assert [s.name for s in specs] == ["a", "b"] == reg.names()


@pytest.mark.parametrize(
    "batch",
    [
        lambda: [build_spec(b), build_spec(b, name="dup"), build_spec(b, name="dup")],
        lambda: [build_spec(b), build_spec(a)],
        lambda: [build_spec(b), build_spec(counts)],
    ],
    ids=["repeated-in-batch", "already-registered", "not-portable"],
)
def test_add_many_is_all_or_nothing(batch):
    reg = Registry()
    reg.add(build_spec(a))
    with pytest.raises((DuplicateTool, SchemaNotPortable)):
        reg.add_many(batch())
    assert reg.names() == ["a"]


def test_add_many_accepts_non_portable_specs_that_opt_out():
    reg = Registry()
    reg.add_many([build_spec(counts, portable=False)])
    assert reg.names() == ["counts"]


def test_for_target_keeps_untargeted_and_matching_tools():
    reg = Registry()
    reg.add(build_spec(a))
    reg.add(build_spec(b, targets=["mcp"]))
    reg.add(build_spec(b, name="c", targets=["openai", "mcp"]))
    assert reg.for_target("openai").names() == ["a", "c"]
    assert reg.for_target("mcp").names() == ["a", "b", "c"]
    assert reg.for_target("custom").names() == ["a"]
    assert reg.for_target("mcp").get("b") is reg.get("b")


def test_exported_to_and_check_target():
    spec = build_spec(a, targets="mcp")
    assert exported_to(spec, "mcp") and not exported_to(spec, "openai")
    assert exported_to(build_spec(a), "anything")
    check_target(spec, "mcp")
    with pytest.raises(ToolUsageError, match=r"not exported to 'openai' \(targets: \['mcp'\]\)"):
        check_target(spec, "openai")
