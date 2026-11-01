import pytest

from agent_adapter.core.errors import ToolDefinitionError
from agent_adapter.core.tool import build_spec


def ping(x: int) -> int:
    """Echo."""
    return x


def test_tags_string_becomes_single_tag():
    assert build_spec(ping, tags="dt").tags == frozenset({"dt"})


def test_valid_policy_is_stored():
    rule = lambda call, ctx: True  # noqa: E731
    spec = build_spec(
        ping,
        kind="action",
        tags=("a", "b"),
        parallel_safe=True,
        once_per_run=True,
        confirm=rule,
        timeout=2,
        retries=3,
        cache_ttl=1.5,
        prompt_hint="When pinging.",
        portable=False,
    )
    assert (spec.kind, spec.tags, spec.confirm, spec.timeout, spec.retries) == (
        "action",
        frozenset({"a", "b"}),
        rule,
        2,
        3,
    )
    assert (spec.once_per_run, spec.cache_ttl, spec.prompt_hint, spec.portable) == (
        True,
        1.5,
        "When pinging.",
        False,
    )


def test_none_is_allowed_for_optional_numbers_and_hint():
    spec = build_spec(ping, timeout=None, cache_ttl=None, prompt_hint=None)
    assert (spec.timeout, spec.cache_ttl, spec.prompt_hint) == (None, None, None)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("kind", ""),
        ("kind", 3),
        ("tags", 5),
        ("tags", ["ok", 1]),
        ("tags", ""),
        ("tags", ["ok", ""]),
        ("timeout", float("inf")),
        ("timeout", float("nan")),
        ("cache_ttl", float("inf")),
        ("cache_ttl", float("nan")),
        ("parallel_safe", "yes"),
        ("once_per_run", 1),
        ("portable", None),
        ("confirm", "always"),
        ("timeout", 0),
        ("timeout", -1.0),
        ("timeout", True),
        ("timeout", "5"),
        ("cache_ttl", 0),
        ("retries", -1),
        ("retries", 1.5),
        ("retries", True),
        ("prompt_hint", 3),
    ],
)
def test_bad_policy_value_is_rejected(option, value):
    with pytest.raises(ToolDefinitionError, match=f"option {option!r}"):
        build_spec(ping, **{option: value})


def test_bad_name_and_description_types_are_rejected():
    with pytest.raises(ToolDefinitionError, match="name"):
        build_spec(ping, name=3)
    with pytest.raises(ToolDefinitionError, match="description"):
        build_spec(ping, description=3)
