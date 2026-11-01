from collections.abc import Iterator
from contextlib import contextmanager

import pytest

import agent_adapter
from agent_adapter import Registry
from agent_adapter.core.errors import DuplicateTool, SchemaNotPortable, ToolNotFound


def make_registry() -> Registry:
    reg = Registry()

    @reg.tool(kind="retrieval", tags=["dt"], prompt_hint="Statute questions.")
    async def sections(query: str, top_k: int = 5) -> list[str]:
        """Fetch sections.

        Args:
            query: What to look for.
        """
        return [query] * top_k

    def legacy_news(text: str, limit: int) -> list[str]:
        """Fetch news."""
        return [text] * limit

    reg.wrap(legacy_news, name="news", fixed={"limit": 1}, tags=["idt"])
    return reg


def test_decorator_registers_and_returns_function_unchanged():
    reg = Registry()

    @reg.tool
    async def ping(x: int) -> int:
        """Echo."""
        return x

    assert "ping" in reg
    assert ping.__agent_tool__ is reg.get("ping")


async def test_decorated_function_is_still_directly_callable():
    reg = Registry()

    @reg.tool
    async def double(x: int) -> int:
        """Double."""
        return x * 2

    assert await double(4) == 8


def test_wrap_registers_existing_function_without_fixed_args_in_schema():
    assert set(make_registry().get("news").json_schema()["properties"]) == {"text"}


async def test_wrapped_tool_invokes_original_function():
    assert await make_registry().get("news").invoke({"text": "gst"}) == ["gst"]


def test_duplicate_name_raises():
    reg = make_registry()

    def sections(q: str) -> str:
        """Duplicate."""
        return q

    with pytest.raises(DuplicateTool):
        reg.wrap(sections)


def test_get_unknown_tool_lists_known_names():
    with pytest.raises(ToolNotFound, match="news"):
        make_registry().get("nope")


def test_non_portable_signature_is_rejected_at_registration():
    def bad(options: dict[str, int]) -> None:
        """Bad."""

    with pytest.raises(SchemaNotPortable, match="open map"):
        Registry().wrap(bad)


def test_portable_false_skips_the_check():
    def bad(options: dict[str, int]) -> None:
        """Bad."""

    reg = Registry()
    reg.wrap(bad, portable=False)
    assert "bad" in reg


def test_iteration_length_and_names_follow_registration_order():
    reg = make_registry()
    assert len(reg) == 2
    assert reg.names() == ["sections", "news"]
    assert [spec.name for spec in reg] == ["sections", "news"]


def test_subset_filters_by_tags_and_kinds():
    reg = make_registry()
    assert reg.subset(tags=["dt"]).names() == ["sections"]
    assert reg.subset(kinds=["tool"]).names() == ["news"]
    assert reg.subset(names=["news"]).names() == ["news"]


def test_subset_combined_filters_use_and_semantics():
    reg = make_registry()
    assert reg.subset(tags=["dt", "idt"], kinds=["retrieval"]).names() == ["sections"]
    assert reg.subset(tags=["idt"], kinds=["retrieval"]).names() == []


def test_subset_specs_are_the_same_objects():
    reg = make_registry()
    assert reg.subset(names=["news"]).get("news") is reg.get("news")


def test_remove_returns_the_spec_and_frees_the_name():
    reg = make_registry()
    spec = reg.get("news")
    assert reg.remove("news") is spec
    assert "news" not in reg and reg.names() == ["sections"]
    reg.add(spec)
    assert reg.names() == ["sections", "news"]


def test_remove_unknown_name_raises():
    with pytest.raises(ToolNotFound, match="ghost"):
        make_registry().remove("ghost")


def test_subset_unknown_name_raises():
    with pytest.raises(ToolNotFound):
        make_registry().subset(names=["ghost"])


def test_catalogue_lists_tools_hints_and_arguments():
    text = make_registry().catalogue()
    assert "### sections\nFetch sections." in text
    assert "When to use: Statute questions." in text
    assert "- `query` (required): What to look for." in text
    assert "- `top_k` (optional)" in text
    assert "### news" in text


@contextmanager
def default_registry_cleanup() -> Iterator[None]:
    """Remove whatever a block registers in the package-level default registry."""
    before = set(agent_adapter.default_registry.names())
    try:
        yield
    finally:
        for name in set(agent_adapter.default_registry.names()) - before:
            agent_adapter.default_registry.remove(name)


def test_package_level_tool_uses_default_registry_and_is_cleaned_up():
    before = agent_adapter.default_registry.names()
    with default_registry_cleanup():

        @agent_adapter.tool(name="pkg_level_echo")
        async def echo(x: int) -> int:
            """Echo."""
            return x

        assert "pkg_level_echo" in agent_adapter.default_registry
    assert agent_adapter.default_registry.names() == before
