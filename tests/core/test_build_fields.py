import functools
import warnings
from collections.abc import Callable
from typing import Annotated, Optional

import pytest
from pydantic import Field

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolArgsInvalid, ToolDefinitionError
from agent_adapter.core.tool import build_spec


def props(spec):
    return spec.json_schema()["properties"]


def test_annotated_description_is_kept():
    def find(q: Annotated[str, Field(description="the query")]) -> str:
        """Find."""
        return q

    assert props(build_spec(find))["q"]["description"] == "the query"


def test_annotated_description_wins_over_docstring():
    def find(q: Annotated[str, Field(description="from annotated")], n: int = 1) -> str:
        """Find.

        Args:
            q: from docstring
            n: how many
        """
        return q

    schema = props(build_spec(find))
    assert schema["q"]["description"] == "from annotated"
    assert schema["n"]["description"] == "how many"


def test_annotated_constraints_are_kept():
    def find(n: Annotated[int, Field(ge=1, le=10)] = 3) -> int:
        """Find."""
        return n

    assert props(build_spec(find))["n"] == {
        "default": 3,
        "maximum": 10,
        "minimum": 1,
        "title": "N",
        "type": "integer",
    }


def test_field_default_without_value_is_required_and_described():
    def find(q: str = Field(description="as default")) -> str:
        """Find."""
        return q

    schema = build_spec(find).json_schema()
    assert schema["required"] == ["q"]
    assert schema["properties"]["q"]["description"] == "as default"


def test_field_default_falls_back_to_docstring_description():
    def find(q: str = Field("x")) -> str:
        """Find.

        Args:
            q: doc text
        """
        return q

    assert props(build_spec(find))["q"]["description"] == "doc text"


async def test_field_default_value_and_constraint_are_honoured():
    def find(n: int = Field(5, ge=1)) -> int:
        """Find."""
        return n

    spec = build_spec(find)
    assert await spec.invoke({}) == 5
    assert await spec.invoke({"n": 7}) == 7
    with pytest.raises(ToolArgsInvalid, match="greater than or equal to 1"):
        await spec.invoke({"n": 0})


async def test_field_default_factory_is_honoured():
    def find(items: list[str] = Field(default_factory=list)) -> list[str]:  # noqa: B008
        """Find."""
        return items

    assert await build_spec(find).invoke({}) == []


def test_hidden_parameter_with_field_default_is_rejected():
    def find(q: str, n: int = Field(5)) -> str:
        """Find."""
        return q

    with pytest.raises(ToolDefinitionError, match="fixed="):
        build_spec(find, hide=["n"])


def f(a: int, b: str) -> str:
    """F."""
    return f"{a}-{b}"


def test_rename_collision_is_rejected():
    with pytest.raises(ToolDefinitionError, match="'b' is used twice"):
        build_spec(f, rename={"a": "b"})


def test_rename_swap_is_allowed():
    spec = build_spec(f, rename={"a": "b", "b": "a"})
    assert spec.llm_to_python == {"b": "a", "a": "b"}


@pytest.mark.parametrize(
    ("new_name", "message"),
    [
        ("a-b", "letters, digits"),
        ("9a", "letters, digits"),
        ("x" * 65, "letters, digits"),
        ("_x", "must not start with '_'"),
        ("model_config", "pydantic"),
        ("model_anything", "pydantic"),
        ("schema", "pydantic"),
        ("copy", "pydantic"),
        ("json", "pydantic"),
        ("dict", "pydantic"),
    ],
)
def test_bad_llm_facing_names_are_rejected(new_name, message):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ToolDefinitionError, match=message):
            build_spec(f, rename={"a": new_name})


def test_bad_parameter_name_is_rejected_without_warnings():
    namespace: dict[str, object] = {}
    exec("def g(model_dump: int) -> int:\n    'G.'\n    return 1", namespace)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ToolDefinitionError, match="pydantic"):
            build_spec(namespace["g"])


def test_underscore_parameter_is_rejected_unless_hidden():
    def g(_internal: int = 1, x: int = 2) -> int:
        """G."""
        return x

    with pytest.raises(ToolDefinitionError, match="must not start with '_'"):
        build_spec(g)
    assert set(props(build_spec(g, hide=["_internal"]))) == {"x"}


def test_rename_of_fixed_parameter_is_rejected():
    with pytest.raises(ToolDefinitionError, match="no effect"):
        build_spec(f, fixed={"a": 1}, rename={"a": "z"})


def test_alias_on_parameter_is_rejected():
    def g(q: Annotated[str, Field(alias="query")]) -> str:
        """G."""
        return q

    with pytest.raises(ToolDefinitionError, match="rename="):
        build_spec(g)


def test_unsupported_type_raises_tool_definition_error():
    class Opaque:
        pass

    def g(x: Opaque) -> int:
        """G."""
        return 1

    with pytest.raises(ToolDefinitionError, match="argument model"):
        build_spec(g)


class MyCtx(ToolContext):
    pass


def test_context_subclass_is_detected():
    def g(x: int, ctx: MyCtx | None = None) -> int:
        """G."""
        return x

    spec = build_spec(g)
    assert spec.context_param == "ctx"
    assert set(props(spec)) == {"x"}


def test_typing_optional_context_is_detected():
    def g(x: int, ctx: Optional[ToolContext] = None) -> int:  # noqa: UP045
        """G."""
        return x

    assert build_spec(g).context_param == "ctx"


def test_second_context_parameter_is_rejected():
    def g(a: ToolContext, b: MyCtx) -> int:
        """G."""
        return 1

    with pytest.raises(ToolDefinitionError, match="more than one ToolContext"):
        build_spec(g)


def test_context_parameter_cannot_be_fixed():
    def g(x: int, ctx: ToolContext | None = None) -> int:
        """G."""
        return x

    with pytest.raises(ToolDefinitionError, match="ToolContext"):
        build_spec(g, fixed={"ctx": None})


def test_positional_only_parameter_is_rejected():
    def g(x: int, /) -> int:
        """G."""
        return x

    with pytest.raises(ToolDefinitionError, match="positional-only"):
        build_spec(g)


def test_unannotated_self_explains_bound_methods():
    def method(self, x: int) -> int:
        """M."""
        return x

    with pytest.raises(ToolDefinitionError, match=r"registry\.wrap\(obj\.method\)"):
        build_spec(method)


class Greeter:
    """Greet someone."""

    async def __call__(self, who: str) -> str:
        return f"hi {who}"


async def test_callable_object_tool():
    spec = build_spec(Greeter(), name="greet")
    assert spec.description == "Greet someone."
    assert set(props(spec)) == {"who"}
    assert await spec.invoke({"who": "x"}) == "hi x"


def test_callable_object_without_name_needs_one():
    with pytest.raises(ToolDefinitionError, match="name="):
        build_spec(Greeter())


async def test_partial_tool_uses_wrapped_function_name_and_docstring():
    def rate(item: str, country: str) -> str:
        """Return the rate.

        Args:
            item: The item.
        """
        return f"{item}/{country}"

    spec = build_spec(functools.partial(rate, country="IN"))
    assert (spec.name, spec.description) == ("rate", "Return the rate.")
    assert set(props(spec)) == {"item"}
    assert props(spec)["item"]["description"] == "The item."
    assert spec.fixed == {"country": "IN"}
    assert await spec.invoke({"item": "steel"}) == "steel/IN"
    with pytest.raises(ToolArgsInvalid, match="country"):
        await spec.invoke({"item": "steel", "country": "US"})


async def test_nested_partial_keywords_are_all_fixed():
    def rate(item: str, country: str, year: int) -> str:
        """Return the rate."""
        return f"{item}/{country}/{year}"

    inner = functools.partial(rate, country="IN")
    outer = functools.partial(inner, year=2025)
    spec = build_spec(outer)
    assert set(props(spec)) == {"item"}
    assert await spec.invoke({"item": "x"}) == "x/IN/2025"


def test_partial_keyword_and_fixed_for_the_same_name_is_rejected():
    def rate(item: str, country: str) -> str:
        """Return the rate."""
        return item

    with pytest.raises(ToolDefinitionError, match="'country' is bound by the partial"):
        build_spec(functools.partial(rate, country="IN"), fixed={"country": "US"})


def test_partial_binding_the_context_parameter_is_rejected():
    def g(x: int, ctx: ToolContext | None = None) -> int:
        """G."""
        return x

    with pytest.raises(ToolDefinitionError, match="'ctx' is the ToolContext parameter"):
        build_spec(functools.partial(g, ctx=ToolContext()))


def test_hide_accepts_a_single_name_as_str():
    def g(q: str, debug: bool = False) -> str:
        """G."""
        return q

    assert set(props(build_spec(g, hide="debug"))) == {"q"}


def test_callable_parameter_without_json_schema_is_rejected():
    def g(cb: Callable[[int], int]) -> int:
        """G."""
        return cb(1)

    with pytest.raises(ToolDefinitionError, match="argument model"):
        build_spec(g)


def test_docstring_same_indent_continuation_joins_parameter():
    def g(query: str) -> str:
        """G.

        Args:
            query: The text, for example
            note: this is still part of query.
        """
        return query

    assert props(build_spec(g))["query"]["description"] == (
        "The text, for example note: this is still part of query."
    )
