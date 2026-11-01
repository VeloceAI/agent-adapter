import json
import threading

import pytest
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolArgsInvalid
from agent_adapter.core.tool import build_spec


class Filter(BaseModel):
    court: str = "sc"
    year: int | None = 2020


async def search(query: str, top_k: int = 8, flt: Filter | None = None) -> dict:
    """Search."""
    return {"query": query, "top_k": top_k, "flt": flt}


async def test_invoke_async_function_with_coerced_args():
    result = await build_spec(search).invoke({"query": "80C", "top_k": "3"})
    assert result == {"query": "80C", "top_k": 3, "flt": None}


async def test_invoke_sync_function_runs_in_worker_thread():
    def where(x: int) -> int:
        """Return the current thread id."""
        return threading.get_ident()

    assert await build_spec(where).invoke({"x": 1}) != threading.get_ident()


async def test_invoke_missing_required_arg_raises():
    with pytest.raises(ToolArgsInvalid) as info:
        await build_spec(search).invoke({"top_k": 1})
    assert info.value.tool == "search"


async def test_invoke_unknown_arg_raises():
    with pytest.raises(ToolArgsInvalid):
        await build_spec(search).invoke({"query": "q", "colour": "red"})


async def test_invoke_merges_fixed_and_maps_renamed_args():
    spec = build_spec(search, fixed={"top_k": 2}, rename={"query": "text"})
    assert await spec.invoke({"text": "q"}) == {"query": "q", "top_k": 2, "flt": None}


async def test_invoke_injects_given_or_default_context():
    async def who(ctx: ToolContext) -> str:
        """Return the run id."""
        return ctx.run_id

    spec = build_spec(who)
    assert await spec.invoke({}, ToolContext(run_id="r1")) == "r1"
    assert await spec.invoke({}) == ""


async def test_invoke_applies_result_transform():
    spec = build_spec(search, result=lambda r: r["query"].upper())
    assert await spec.invoke({"query": "q"}) == "Q"


async def test_invoke_turns_strict_mode_nulls_into_defaults():
    out = await build_spec(search).invoke(
        {"query": "q", "top_k": None, "flt": {"court": None, "year": None}}
    )
    assert out["top_k"] == 8
    assert out["flt"] == Filter(court="sc", year=2020)


async def test_invoke_passes_nested_models_as_instances():
    out = await build_spec(search).invoke({"query": "q", "flt": {"court": "hc"}})
    assert isinstance(out["flt"], Filter)
    assert out["flt"].court == "hc"


async def test_invoke_does_not_mutate_caller_args():
    args = {"query": "q", "top_k": None}
    await build_spec(search).invoke(args)
    assert args == {"query": "q", "top_k": None}


async def test_strict_null_means_default_even_when_the_type_accepts_none():
    def pick(x: int | None = 5, y: int | None = None) -> tuple[int | None, int | None]:
        """Pick."""
        return x, y

    assert await build_spec(pick).invoke({"x": None, "y": None}) == (5, None)


async def test_strict_null_for_a_required_field_is_still_invalid():
    with pytest.raises(ToolArgsInvalid, match="query"):
        await build_spec(search).invoke({"query": None})


class Aliased(BaseModel):
    the_val: int = Field(3, alias="theVal")


class ByName(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    the_val: int = Field(3, alias="theVal")


async def test_strict_nulls_follow_nested_aliases():
    async def get(n: Aliased, m: ByName, items: list[Aliased]) -> list[int]:
        """Get."""
        return [n.the_val, m.the_val, *(item.the_val for item in items)]

    args = {"n": {"theVal": None}, "m": {"the_val": None}, "items": [{"theVal": None}]}
    assert await build_spec(get).invoke(args) == [3, 3, 3]


class Positive(BaseModel):
    x: int

    @field_validator("x")
    @classmethod
    def positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be positive")
        return v


async def test_args_invalid_errors_are_json_safe_and_carry_no_input():
    async def take(m: Positive, secret: int) -> int:
        """Take."""
        return m.x

    with pytest.raises(ToolArgsInvalid) as info:
        await build_spec(take).invoke({"m": {"x": -1}, "secret": "card-4111-1111"})
    err = info.value
    json.dumps(err.errors)
    assert all(set(e) <= {"type", "loc", "msg", "ctx"} for e in err.errors)
    assert err.errors[0]["loc"] == ["m", "x"]
    assert err.errors[0]["ctx"] == {"error": "must be positive"}
    assert "m.x: Value error, must be positive" in str(err)
    assert "card-4111" not in str(err) and "card-4111" not in json.dumps(err.errors)
