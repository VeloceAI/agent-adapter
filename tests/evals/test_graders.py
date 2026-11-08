import logging
import math

import pytest
from pydantic import BaseModel

from agent_adapter.core.errors import ConfigError
from agent_adapter.evals import graders as evals_graders
from agent_adapter.evals.graders import (
    all_of,
    code_check,
    contains,
    exact,
    json_schema,
    max_cost,
    max_latency,
    regex,
    tools_called,
)
from agent_adapter.evals.model import Score
from agent_adapter.optimiser import CaseRun
from agent_adapter.optimiser import graders as optimiser_graders
from tests.evals.fakes import case, run


def test_optimiser_graders_are_the_evals_graders():
    for name in optimiser_graders.__all__:
        assert getattr(optimiser_graders, name) is getattr(evals_graders, name)


def test_graders_carry_their_names():
    assert exact().__name__ == "exact"
    assert tools_called("a").__name__ == "tools_called"
    assert all_of(exact()).__name__ == "all_of"


async def test_score_reason_is_its_notes():
    assert Score(value=0.5, notes="why").reason == "why"


async def test_text_graders_read_non_text_outputs_as_json():
    assert (await exact()(case(expected='{"a": 1}'), run({"a": 1}))).passed
    assert (await contains()(case(expected="1"), run([1, 2]))).passed


async def test_regex_with_a_fixed_pattern_ignores_expected():
    grade = regex(r"ship+ed")
    assert (await grade(case(expected=None), run("SHIPPED today"))).passed
    miss = await grade(case(), run("lost"))
    assert miss.passed is False and miss.value == 0.0


def test_regex_refuses_a_bad_pattern():
    with pytest.raises(ConfigError, match="invalid pattern"):
        regex("(")


class Order(BaseModel):
    id: str
    qty: int


SCHEMA = {
    "type": "object",
    "properties": {"id": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}},
    "required": ["id", "qty"],
}


@pytest.mark.parametrize(
    ("output", "passed"),
    [
        ('{"id": "a", "qty": 2}', True),
        ({"id": "a", "qty": 2}, True),
        (Order(id="a", qty=3), True),
        ('{"id": "a", "qty": 0}', False),
        ({"id": "a"}, False),
        ("not json", False),
        (None, False),
    ],
)
async def test_json_schema(output, passed):
    score = await json_schema(SCHEMA)(case(), run(output))
    assert score.passed is passed
    assert score.notes == ("" if passed else score.notes) and (passed or score.notes)


def test_json_schema_needs_an_object():
    with pytest.raises(ConfigError, match="schema object"):
        json_schema(["type"])  # type: ignore[arg-type]


async def test_tools_called_scores_the_share_called():
    grade = tools_called("order_status", "refund")
    both = await grade(case(), run(tool_calls=("order_status", "refund", "order_status")))
    half = await grade(case(), run(tool_calls=("order_status",)))
    assert both.passed and both.value == 1.0
    assert half.passed is False and half.value == 0.5 and "not called: refund" in half.notes


async def test_tools_called_forbid_and_only_score_zero():
    forbidden = await tools_called(forbid=["refund"])(case(), run(tool_calls=("refund",)))
    assert forbidden.value == 0.0 and "forbidden: refund" in forbidden.notes
    extra = await tools_called("order_status", only=True)(
        case(), run(tool_calls=("order_status", "weather"))
    )
    assert extra.passed is False and "not allowed: weather" in extra.notes
    clean = await tools_called(forbid=["refund"])(case(), run(tool_calls=()))
    assert clean.passed and clean.value == 1.0


async def test_tools_called_reads_optimiser_metrics():
    case_run = CaseRun(
        case_id="c1",
        label="",
        ok=True,
        delivered=True,
        output="x",
        transcript="",
        metrics={"tool_calls.order_status": 1.0, "tool_calls.refund": 0.0},
    )
    assert (await tools_called("order_status", forbid=["refund"])(case(), case_run)).passed


async def test_tools_called_without_any_record_is_ungraded(caplog):
    caplog.set_level(logging.WARNING)

    class Bare:
        output = "x"

    assert await tools_called("a")(case(), Bare()) is None
    assert "no tool calls recorded" in caplog.text


@pytest.mark.parametrize("bad", [(), ("",), (3,)])
def test_tools_called_refuses_bad_names(bad):
    with pytest.raises(ConfigError):
        tools_called(*bad)


async def test_max_cost_and_latency_check_their_limits():
    cheap, dear = run(cost_usd=0.001), run(cost_usd=0.5)
    assert (await max_cost(0.01)(case(), cheap)).passed
    over = await max_cost(0.01)(case(), dear)
    assert over.passed is False and "over" in over.notes
    assert (await max_latency(1)(case(), run(latency_s=0.2))).passed
    assert (await max_latency(1)(case(), run(latency_s=2.5))).passed is False


async def test_limits_read_optimiser_metrics_or_stay_ungraded(caplog):
    caplog.set_level(logging.WARNING)
    case_run = CaseRun(
        case_id="c1",
        label="",
        ok=True,
        delivered=True,
        output="",
        transcript="",
        metrics={"cost_usd": 0.002},
    )
    assert (await max_cost(0.01)(case(), case_run)).passed
    assert await max_latency(1)(case(), case_run) is None
    assert "has no latency_s" in caplog.text


@pytest.mark.parametrize("bad", [-1, math.inf, math.nan, True, "1"])
def test_limits_refuse_bad_values(bad):
    with pytest.raises(ConfigError, match="finite number"):
        max_cost(bad)
    with pytest.raises(ConfigError, match="finite number"):
        max_latency(bad)


async def test_code_check_rejects_odd_results_without_logging_them(caplog):
    caplog.set_level(logging.WARNING)
    assert await code_check(lambda c, r: "secret-value")(case(), run()) is None
    assert "secret-value" not in caplog.text and "returned str" in caplog.text


async def test_all_of_mixes_new_graders():
    grade = all_of(contains(), tools_called("order_status"), weights=[1, 3])
    score = await grade(case(), run("shipped", tool_calls=()))
    assert score.value == pytest.approx(0.25) and score.passed is False
