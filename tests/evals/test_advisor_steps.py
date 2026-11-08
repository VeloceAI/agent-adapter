"""Steps from structured fields, `StepModels` routing and the advisor's worst-case spend cap."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent_adapter.core.context import NODE_KEY, ToolContext
from agent_adapter.core.errors import BudgetExceeded, ConfigError, ModelError
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message, StreamDone, StreamEvent
from agent_adapter.core.types import LedgerEntry, Usage
from agent_adapter.cost.prices import Prices
from agent_adapter.evals._spend import INPUT_MARGIN_TOKENS, CappedModel, SpendCap, cap_graders
from agent_adapter.evals.advisor import (
    StepModels,
    apply_mapping,
    current_step,
    entry_step,
    step_of,
)
from agent_adapter.evals.graders import all_of, contains
from agent_adapter.evals.judge import LLMJudge
from agent_adapter.models.callable import CallableModel
from agent_adapter.runtime.current import use_context

PRICES = Prices.empty("t").merged(
    {
        "version": "t",
        "models": {
            "a": {"provider": "p", "input": 1.0, "output": 1.0},
            "w": {"provider": "p", "input": 1.0, "cache_write": 2.0, "output": 1.0},
        },
    }
)


def model(name: str, tokens: int = 10) -> CallableModel:
    """A model that answers with its own name and the model it was asked for."""
    return CallableModel(
        lambda r: {
            "text": f"{name}|{r.model}|{r.max_output_tokens}",
            "usage": {"input_tokens": tokens, "output_tokens": tokens},
        },
        provider="p",
        model=name,
    )


def request(stage: str = "", **fields: Any) -> LLMRequest:
    return LLMRequest(messages=(Message(role="user", content="hi"),), stage=stage, **fields)


def in_node(node: str) -> ToolContext:
    return ToolContext(run_id="r", call_id="x", extras={NODE_KEY: node})


@pytest.mark.parametrize(
    ("stage", "node", "step"),
    [
        ("triage", "triage", "node:triage"),
        ("tool_loop", "reply", "node:reply"),
        ("tool_loop", "", "stage:tool_loop"),
        ("", "", "stage:llm"),
    ],
)
def test_step_of_names_nodes_and_stages(stage: str, node: str, step: str):
    assert step_of(stage, node) == step


def test_entry_step_reads_the_node_and_stage_fields():
    entry = LedgerEntry(run_id="r", call_id="c", kind="llm", stage="s", metadata={"node": "n"})
    assert entry_step(entry) == "node:n"
    assert entry_step(entry.model_copy(update={"metadata": {}})) == "stage:s"


def test_current_step_reads_the_running_node():
    assert current_step(request("draft")) == "stage:draft"
    with use_context(in_node("triage")):
        assert current_step(request("draft")) == "node:triage"


async def test_step_models_route_by_step_and_clear_the_model():
    router = StepModels(model("base"), {"node:triage": model("cheap")})
    with use_context(in_node("triage")):
        routed = await router.generate(request("t", model="pinned"))
    with use_context(in_node("reply")):
        kept = await router.generate(request("t", model="pinned"))
    assert routed.text.startswith("cheap|None")
    assert kept.text.startswith("base|pinned")
    assert router.model == "base" and router.provider == "steps"


async def test_step_models_star_covers_unlisted_steps_and_streams():
    router = StepModels(model("base"), {"*": model("cheap"), "stage:keep": model("base2")})
    events = [e async for e in router.stream(request("other", stream=True))]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].response.text.startswith("cheap")
    assert router.port_for("stage:keep").model == "base2"


def test_step_models_cache_key_names_ports_and_node():
    router = StepModels(model("base"), {"node:a": model("cheap")})
    with use_context(in_node("a")):
        fields = router.cache_key_fields()
    assert fields["node"] == "a"
    assert fields["steps"]["node:a"]["model"] == "cheap"
    assert router.cache_key_fields()["node"] == ""


@pytest.mark.parametrize(
    "steps", [{"": model("x")}, {"node:a": object()}], ids=["empty-key", "not-a-port"]
)
def test_step_models_reject_bad_steps(steps: Any):
    with pytest.raises(ConfigError):
        StepModels(model("base"), steps)


def test_apply_mapping_builds_the_router_and_rejects_unknown_names():
    ports = {"base": model("base"), "cheap": model("cheap")}
    router = apply_mapping(ports["base"], {"node:a": "cheap"}, ports)
    assert router.port_for("node:a") is ports["cheap"]
    with pytest.raises(ConfigError, match="unknown candidates"):
        apply_mapping(ports["base"], {"node:a": "missing"}, ports)


def test_the_worst_case_is_every_request_byte_in_and_the_output_limit_out():
    cap = SpendCap(1.0, PRICES, max_output_tokens=100)
    asked = request(max_output_tokens=50)
    size = len(asked.model_dump_json().encode("utf-8")) + INPUT_MARGIN_TOKENS
    assert cap.worst_usd("a", asked) == pytest.approx((size + 50) / 1e6)
    unlimited = request()
    size = len(unlimited.model_dump_json().encode("utf-8")) + INPUT_MARGIN_TOKENS
    assert cap.worst_usd("a", unlimited) == pytest.approx((size + 100) / 1e6)
    assert cap.worst_usd("w", unlimited) == pytest.approx((2 * size + 100) / 1e6)  # cache write
    ttl = Prices.empty("t").merged(
        {
            "version": "t",
            "models": {
                "h": {"provider": "p", "input": 1, "output": 1, "cache_write_by_ttl": {"1h": 3}}
            },
        }
    )
    assert SpendCap(1.0, ttl, max_output_tokens=100).worst_usd("h", unlimited) == pytest.approx(
        (3 * size + 100) / 1e6
    )
    assert cap.limited(request(max_output_tokens=500)).max_output_tokens == 100
    assert cap.limited(asked) is asked
    assert SpendCap(None, PRICES).limited(unlimited) is unlimited


async def test_capped_calls_are_sent_with_the_output_limit():
    capped = CappedModel(model("a"), SpendCap(1.0, PRICES, max_output_tokens=64))
    assert (await capped.generate(request())).text == "a|None|64"


async def test_the_cap_holds_the_worst_case_and_is_never_exceeded():
    one = SpendCap(1.0, PRICES, max_output_tokens=100).worst_usd("a", request())
    cap = SpendCap(one * 1.005, PRICES, max_output_tokens=100)
    capped = CappedModel(model("a"), cap)
    await capped.generate(request())  # holds `one`, settles the real 20 tokens
    assert cap.spent_usd == pytest.approx(20 / 1e6)
    with pytest.raises(BudgetExceeded):  # 20 tokens spent + another worst case > the cap
        await capped.generate(request())
    assert cap.stopped and cap.spent_usd <= one * 1.005
    with pytest.raises(BudgetExceeded):  # once stopped, every call is refused
        await capped.generate(request(max_output_tokens=1))


async def test_concurrent_holds_count_together():
    one = SpendCap(1.0, PRICES, max_output_tokens=100).worst_usd("a", request())
    cap = SpendCap(one * 1.5, PRICES, max_output_tokens=100)
    first = await cap.reserve("a", request())
    with pytest.raises(BudgetExceeded):
        await cap.reserve("a", request())
    await cap.settle(first, None, "a")  # no reply: the whole hold counts as spent
    assert cap.spent_usd == pytest.approx(one)


async def test_capped_model_streams_and_settles():
    cap = SpendCap(1.0, PRICES)
    events = [e async for e in CappedModel(model("a"), cap).stream(request(stream=True))]
    assert isinstance(events[-1], StreamDone)
    assert cap.spent_usd == pytest.approx(20 / 1e6)


class Broken:
    provider = "p"
    model = "a"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise ModelError("down", provider="p")

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        raise ModelError("down", provider="p")
        yield  # pragma: no cover

    def cache_key_fields(self) -> dict[str, Any]:
        return {"k": 1}


async def test_a_failed_call_counts_its_whole_hold():
    cap = SpendCap(1.0, PRICES, max_output_tokens=10)
    capped = CappedModel(Broken(), cap)
    with pytest.raises(ModelError):
        await capped.generate(request())
    with pytest.raises(ModelError):
        _ = [e async for e in capped.stream(request())]
    assert cap.spent_usd == pytest.approx(2 * cap.worst_usd("a", request()))
    assert capped.cache_key_fields() == {"k": 1}


async def test_a_reply_dearer_than_its_hold_is_logged(caplog: pytest.LogCaptureFixture):
    cap = SpendCap(1.0, PRICES, max_output_tokens=1)
    await CappedModel(model("a", tokens=10_000), cap).generate(request())
    assert "worst-case hold" in caplog.text


async def test_without_a_limit_nothing_is_held_and_no_price_is_needed():
    cap = SpendCap(None, Prices.empty("t"))
    cap.check_priced({"x": model("unpriced")})
    assert await cap.reserve("unpriced", request()) == ("", 0.0)


def test_spend_cap_fails_closed_on_unpriced_or_per_second_models():
    per_second = Prices.empty("t").merged(
        {"version": "t", "models": {"gpu": {"provider": "p", "per_second": 0.01}}}
    )
    with pytest.raises(ConfigError, match="zzz"):
        SpendCap(1.0, PRICES).check_priced({"zzz": model("zzz")})
    with pytest.raises(ConfigError, match="gpu"):
        SpendCap(1.0, per_second).check_priced({"gpu": model("gpu")})


async def test_settle_prices_by_the_port_model_when_the_reply_model_is_unknown():
    cap = SpendCap(1.0, PRICES)
    reply = LLMResponse(
        provider="p",
        model="renamed",
        usage=Usage(input_tokens=10, output_tokens=10),
        message=Message(role="assistant", content=""),
    )
    await cap.settle(await cap.reserve("a", request()), reply, "a")
    assert cap.spent_usd == pytest.approx(20 / 1e6)


async def test_judges_get_capped_models_also_inside_all_of():
    cap = SpendCap(1.0, PRICES)
    judge = LLMJudge(model("a"), "Says pong.")
    plain = contains()
    listed, found = cap_graders([judge, plain, all_of(plain, judge, weights=[1, 2])], cap)
    assert isinstance(listed, list)
    assert isinstance(listed[0], LLMJudge) and isinstance(listed[0].model, CappedModel)
    assert listed[0]._unguarded == listed[0].grade and judge.model.model == "a"
    assert listed[1] is plain
    rebuilt = listed[2]
    assert rebuilt.weights == (1.0, 2.0)  # type: ignore[attr-defined]
    assert isinstance(rebuilt.parts[1].model, CappedModel)  # type: ignore[attr-defined]
    assert sorted(found) == ["judge of grader 0", "judge of grader 2[1]"]
    untouched = all_of(plain)
    named, none = cap_graders({"only": untouched}, cap)
    assert isinstance(named, dict) and named["only"] is untouched and none == {}


async def test_a_reply_without_usage_keeps_its_whole_hold():
    cap = SpendCap(1.0, PRICES, max_output_tokens=10)
    silent = CallableModel(lambda r: "no usage reported", provider="p", model="a")
    await CappedModel(silent, cap).generate(request())
    assert cap.spent_usd == pytest.approx(cap.worst_usd("a", request()))


async def test_a_retried_call_holds_and_counts_each_attempt():
    attempts: list[int] = []

    def flaky(request: LLMRequest) -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) == 1:
            raise ModelError("busy", provider="p")
        return {"text": "ok", "usage": {"input_tokens": 10, "output_tokens": 10}}

    cap = SpendCap(1.0, PRICES, max_output_tokens=10)
    capped = CappedModel(CallableModel(flaky, provider="p", model="a"), cap)
    with pytest.raises(ModelError):
        await capped.generate(request())
    await capped.generate(request())  # the caller's retry is a new call with its own hold
    worst = cap.worst_usd("a", request())
    assert cap.spent_usd == pytest.approx(worst + 20 / 1e6)
