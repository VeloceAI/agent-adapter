import asyncio

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, OptimiserError
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.cost.ledger import Ledger
from agent_adapter.models.callable import CallableModel
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.optimiser import CallableHost, CaseRun, EvalCase, Host, NullModel, optimise
from agent_adapter.optimiser._transcript import TranscriptHook
from agent_adapter.optimiser.graders import contains
from tests.optimiser.fakes import ScriptedProposer, ToyAgent, proposal, tool_reply
from tests.optimiser.loop_support import config

SURFACE = {"prompt/system": "You answer tax questions.", "tool/search/description": "Search."}
CASES = [EvalCase(id="q1", input="tuition fees", expected="80C")]


async def answer(tree, case, model):
    return f"{case.input}: {tree.files['prompt/system']}"


def host(run_case=answer, **kwargs):
    return CallableHost(run_case, surface=SURFACE, cases=CASES, **kwargs)


async def test_a_text_answer_becomes_a_delivered_case_run():
    run = await host().run(host().surface(), CASES[0], run_id="r", model=NullModel())
    assert (run.ok, run.delivered, run.case_id) == (True, True, "q1")
    assert run.output == "tuition fees: You answer tax questions."
    assert run.transcript == f"[USER] tuition fees\n[OUTPUT] {run.output}"


async def test_an_empty_answer_is_not_delivered_and_a_case_run_passes_through():
    async def empty(tree, case, model):
        return "  "

    run = await host(empty).run(host().surface(), CASES[0], run_id="r", model=NullModel())
    assert (run.ok, run.delivered) == (True, False)
    mine = CaseRun(case_id="q1", label="", ok=True, delivered=True, output="x", transcript="t")

    async def own(tree, case, model):
        return mine

    assert await host(own).run(host().surface(), CASES[0], run_id="r", model=NullModel()) is mine


async def test_an_exception_or_a_wrong_type_is_a_failed_run_never_raised():
    async def broken(tree, case, model):
        raise RuntimeError("agent crashed")

    async def wrong(tree, case, model):
        return 42

    for run_case, kind in ((broken, "internal-error"), (wrong, "config-invalid")):
        run = await host(run_case).run(host().surface(), CASES[0], run_id="r", model=NullModel())
        assert (run.ok, run.delivered) == (False, False)
        assert run.error.startswith(kind)


async def test_cancellation_is_not_swallowed():
    async def cancelled(tree, case, model):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await host(cancelled).run(host().surface(), CASES[0], run_id="r", model=NullModel())


async def test_model_calls_are_recorded_and_confirm_tool_arguments_withheld():
    agent = ToyAgent()

    async def calls(tree, case, model):
        await model.generate(LLMRequest(messages=(Message(role="user", content="hi"),), stage="a"))
        return "done"

    async def wipe(request):
        return tool_reply("wipe", {"target": "db"})

    for port, withhold in ((agent.port, ()), (CallableModel(wipe), ("wipe",))):
        model = InstrumentedModel(port, hooks=[TranscriptHook()], ledger=Ledger()).bind(
            ToolContext(run_id="r")
        )
        run = await host(calls, withhold=withhold).run(
            host().surface(), CASES[0], run_id="r", model=model
        )
        assert run.metrics["llm_calls"] == 1.0
        assert "[INPUT] a: hi" in run.transcript
    assert "[ACTION] wipe (arguments withheld)" in run.transcript
    assert "db" not in run.transcript
    assert run.metrics["tool_calls.wipe"] == 1.0


def test_it_is_a_host_and_its_kinds_come_from_the_surface():
    made = host(judge=contains(), domain="tax", caps={"prompt/system": 100})
    assert isinstance(made, Host)
    caps = made.caps()
    assert caps.kinds == frozenset({"prompt", "tool"})
    assert caps.max_chars["prompt/system"] == 100
    assert made.cases() == tuple(CASES) and made.domain == "tax"


def test_validate_guards_required_paths_off_surface_paths_and_runs_yours():
    made = host(required=["prompt/system"], validate=lambda tree: ["mine"])
    tree = made.surface()
    assert made.validate(tree) == ["mine"]
    assert "prompt/system: a required path cannot be deleted" in made.validate(
        tree.with_edits({"prompt/system": None})
    )
    assert "prompt/system: a required path cannot be empty" in made.validate(
        tree.with_edits({"prompt/system": " "})
    )
    assert "prompt/layout/x: not on this host's surface" in made.validate(
        tree.with_edits({"prompt/layout/x": "new"})
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"surface": {"../etc": "x"}}, "not an editable path"),
        ({"surface": {}}, "surface is empty"),
        ({"surface": {"prompt/system": 3}}, "must be text"),
        ({"caps": {"tool/*/name": 3}}, "unknown pattern"),
        ({"caps": {"prompt/system": 0}}, "int >= 1"),
        ({"required": ["prompt/layout/a"]}, "not on the surface"),
    ],
)
def test_bad_construction_raises_config_error(kwargs, message):
    options = {"surface": SURFACE, "cases": CASES, **kwargs}
    with pytest.raises(ConfigError, match=message):
        CallableHost(answer, **options)


def test_not_callable_and_duplicate_cases_are_rejected():
    with pytest.raises(ConfigError, match="callable"):
        CallableHost("nope", surface=SURFACE, cases=CASES)  # type: ignore[arg-type]
    with pytest.raises(OptimiserError, match="duplicate"):
        CallableHost(answer, surface=SURFACE, cases=CASES * 2)


async def test_null_model_always_raises_config_error():
    request = LLMRequest(messages=(Message(role="user", content="x"),))
    model = NullModel()
    assert (model.provider, model.model, model.cache_key_fields()) == ("none", "none", {})
    with pytest.raises(ConfigError, match="without --model"):
        await model.generate(request)
    with pytest.raises(ConfigError):
        async for _ in model.stream(request):
            pass


async def test_a_callable_host_optimises_end_to_end(tmp_path):
    agent = ToyAgent()

    async def ask(tree, case, model):
        messages = (
            Message(role="system", content=tree.files["prompt/system"]),
            Message(role="user", content=str(case.input)),
        )
        return (await model.generate(LLMRequest(messages=messages, stage="agent"))).text

    proposer = ScriptedProposer({"know1": proposal(("prompt/system", "Cite the section."))})
    result = await optimise(
        host(ask, judge=contains()),
        config(tmp_path, lessons_path=""),
        proposer=proposer.port,
        model=agent.port,
    )
    assert result.chosen == "agg1"
    assert (result.base.mean_score, result.probes["agg1"].mean_score) == (0.0, 1.0)


async def test_a_framework_interrupt_is_not_swallowed():
    class GraphBubbleUp(Exception):  # LangGraph's interrupt base, matched by name
        pass

    async def interrupted(tree, case, model):
        raise GraphBubbleUp

    with pytest.raises(GraphBubbleUp):
        await host(interrupted).run(host().surface(), CASES[0], run_id="r", model=NullModel())


async def test_a_returned_case_run_keeps_the_recorded_counts_and_transcript():
    agent = ToyAgent()

    async def own(tree, case, model):
        await model.generate(LLMRequest(messages=(Message(role="user", content="hi"),), stage="a"))
        metrics = {"cost_usd": 0.5, "tool_calls": 7.0}
        return CaseRun(
            case_id="q1",
            label="",
            ok=True,
            delivered=True,
            output="x",
            transcript="",
            metrics=metrics,
        )

    model = InstrumentedModel(agent.port, hooks=[TranscriptHook()], ledger=Ledger()).bind(
        ToolContext(run_id="r")
    )
    run = await host(own).run(host().surface(), CASES[0], run_id="r", model=model)
    assert run.metrics["cost_usd"] == 0.5
    assert run.metrics["llm_calls"] == 1.0 and run.metrics["tool_calls"] == 7.0
    assert "[INPUT] a: hi" in run.transcript
