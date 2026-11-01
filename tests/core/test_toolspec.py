from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolArgsInvalid, ToolUsageError
from agent_adapter.core.tool import build_spec

if TYPE_CHECKING:
    from collections import OrderedDict


class Hit(BaseModel):
    title: str
    score: float


def search(query: str) -> list[Hit]:
    """Search."""
    return [Hit(title=query, score=1.0)]


def test_spec_is_hashable_with_identity_equality():
    a, b = build_spec(search), build_spec(search)
    assert hash(a) == hash(a)
    assert a == a and a != b
    assert {a, b} == {b, a} and len({a, b}) == 2


def test_returns_holds_the_resolved_return_annotation():
    assert build_spec(search).returns == list[Hit]


def test_result_schema_comes_from_the_return_annotation():
    schema = build_spec(search).result_schema()
    assert schema is not None
    assert schema["type"] == "array"
    assert schema["items"]["$ref"] == "#/$defs/Hit"


def test_result_schema_is_none_without_annotation_or_for_none():
    def a(x: int):  # type: ignore[no-untyped-def]
        """A."""

    def b(x: int) -> None:
        """B."""

    assert build_spec(a).returns is None and build_spec(a).result_schema() is None
    assert build_spec(b).returns is None and build_spec(b).result_schema() is None


def test_result_schema_is_none_when_a_result_transform_is_set():
    assert build_spec(search, result=len).result_schema() is None


class Opaque:
    pass


def test_result_schema_is_none_for_types_without_a_schema(caplog):
    def make(x: int) -> Opaque:
        """Make."""
        return Opaque()

    spec = build_spec(make)
    assert spec.returns is Opaque
    assert spec.result_schema() is None
    assert "no result schema" in caplog.text


def test_unresolvable_return_annotation_does_not_fail_registration():
    def recent(n: int) -> OrderedDict[str, int]:
        """Recent."""
        raise NotImplementedError

    spec = build_spec(recent)
    assert spec.returns is None
    assert set(spec.json_schema()["properties"]) == {"n"}


def test_json_schema_is_cached_and_returned_as_a_copy(monkeypatch):
    spec = build_spec(search)
    first = spec.json_schema()
    first["properties"]["query"]["type"] = "mutated"

    def regenerate(*args: object, **kwargs: object) -> None:
        raise AssertionError("schema regenerated")

    monkeypatch.setattr(spec.args_model, "model_json_schema", regenerate)
    assert spec.json_schema()["properties"]["query"]["type"] == "string"


async def test_call_runs_an_already_validated_model():
    async def who(n: int, ctx: ToolContext) -> str:
        """Who."""
        return f"{ctx.run_id}:{n}"

    spec = build_spec(who)
    validated = spec.validate_args({"n": "2"})
    assert await spec.call(validated, ToolContext(run_id="r")) == "r:2"


async def test_call_rejects_a_model_of_another_tool():
    class Other(BaseModel):
        n: int

    with pytest.raises(ToolUsageError, match="expected SearchArgs") as info:
        await build_spec(search).call(Other(n=1))
    assert isinstance(info.value, TypeError)
    assert not isinstance(info.value, ToolArgsInvalid)
