from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from agent_adapter.schema.portable import check_portable


class Court(StrEnum):
    SC = "sc"
    HC = "hc"


class Filter(BaseModel):
    court: Court = Court.SC
    year: int | None = None


class Good(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Text.")
    top_k: int = Field(8, ge=1, le=50)
    mode: Literal["fast", "deep"] = "fast"
    tag: Literal["x"] = "x"
    flt: Filter | None = None
    many: list[Filter] = Field(default_factory=list)
    when: datetime | None = None


class TreeNode(BaseModel):
    children: list[TreeNode] = Field(default_factory=list)


def problems_for(**fields: Any) -> str:
    model = create_model("M", **{name: (annotation, ...) for name, annotation in fields.items()})
    return " | ".join(check_portable(model.model_json_schema()))


def test_portable_schema_has_no_problems():
    assert check_portable(Good.model_json_schema()) == []


def test_root_must_be_object():
    assert "root must be an object" in check_portable({"type": "string"})[0]


def test_dict_is_rejected():
    assert "open map (dict)" in problems_for(options=dict[str, int])


def test_any_is_rejected():
    assert "untyped value (Any)" in problems_for(value=Any)


def test_tuple_is_rejected_once():
    problems = check_portable(create_model("M", pair=(tuple[int, str], ...)).model_json_schema())
    assert problems == ["$.pair: tuple is not portable"]


def obj(**props: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props}


def test_type_list_with_one_type_and_null_is_portable():
    assert check_portable(obj(n={"type": ["integer", "null"]})) == []


def test_type_list_nullable_object_is_still_walked():
    open_map = {"type": ["object", "null"], "additionalProperties": True}
    assert check_portable(obj(m=open_map)) == ["$.m: open map (dict) is not portable"]


def test_type_list_with_several_types_is_rejected():
    for kinds in (["integer", "string"], ["integer", "string", "null"]):
        assert check_portable(obj(v={"type": kinds})) == [
            "$.v: union of several types is not portable"
        ]


def test_type_list_without_a_non_null_type_is_rejected():
    for kinds in (["null"], []):
        assert check_portable(obj(v={"type": kinds})) == [
            "$.v: type list without a non-null type is not portable"
        ]


def test_single_item_all_of_is_allowed():
    assert check_portable(obj(a={"allOf": [{"type": "string"}]})) == []


def test_multi_item_all_of_is_rejected():
    schema = obj(a={"allOf": [{"type": "string"}, {"type": "string"}]})
    assert check_portable(schema) == ["$.a: intersection (allOf) is not portable"]


def test_multi_type_union_is_rejected():
    assert "union of several types" in problems_for(value=int | str)


def test_recursive_model_is_rejected():
    assert "recursive model" in problems_for(tree=TreeNode)


def test_non_string_enum_is_rejected():
    assert "enum values must be strings" in problems_for(level=Literal[1, 2])


def test_pattern_is_rejected():
    assert "regex pattern" in problems_for(code=Annotated[str, Field(pattern=r"^\d+$")])


def test_date_format_is_rejected_but_datetime_allowed():
    assert "format 'date'" in problems_for(day=date)
    assert problems_for(at=datetime) == ""
