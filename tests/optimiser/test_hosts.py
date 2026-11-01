import asyncio
import json

import pytest
from pydantic import BaseModel

from agent_adapter.cache.layout import PromptLayout
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import OptimiserError
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.cost.ledger import Ledger
from agent_adapter.exec import Executor
from agent_adapter.flow import Flow, node_model, node_prompt
from agent_adapter.hooks import BudgetHook, GuardrailHook, LoggingHook, Rate, RateLimitHook
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.optimiser._transcript import TranscriptHook
from agent_adapter.optimiser.hosts import FlowHost, Host, RegistryHost, apply_tree
from agent_adapter.optimiser.model import EvalCase, HarnessTree
from agent_adapter.optimiser.surface import validate_tree
from tests.optimiser.fakes import BlockHook, LoopAgent, make_registry, reply, system_of, user_of

CASE = EvalCase(id="c1", input="rent relief", expected="80C")


def registry_host(**kwargs) -> RegistryHost:
    options = {
        "layout": PromptLayout(static="You answer tax questions.", sections={"style": "Be brief."}),
        "cases": [CASE],
        "hooks": [BudgetHook(5.0)],
        "domain": "tax",
        "allow_safety_tuning": True,
    }
    options.update(kwargs)
    return RegistryHost(make_registry(), **options)


def test_registry_host_is_a_host_and_exposes_its_surface():
    host = registry_host()
    assert isinstance(host, Host)
    assert host.surface().files == {
        "prompt/system": "You answer tax questions.",
        "prompt/layout/style": "Be brief.",
        "tool/search/description": "Search the archive.",
        "tool/fetch/description": "Fetch one document.",
        "tool/fetch/prompt_hint": "Use after search.",
        "tool/delete/description": "Delete a document.",
        "hook/budget/config.json": json.dumps(
            {"max_usd": 5.0, "on_exceed": "stop", "scope": "run"}, indent=2, sort_keys=True
        ),
    }
    caps = host.caps()
    assert caps.kinds == {"tool", "hook", "prompt"}
    assert caps.hook_schemas == {"budget": BudgetHook.config_schema()}
    assert [c.id for c in host.cases()] == ["c1"]
    assert validate_tree(host.surface(), caps, host) == []


def test_registry_host_rejects_duplicate_case_ids_and_hook_names():
    with pytest.raises(OptimiserError, match="duplicate case ids"):
        registry_host(cases=[CASE, CASE])
    with pytest.raises(OptimiserError, match="duplicate hook names"):
        registry_host(hooks=[BudgetHook(1.0), BudgetHook(2.0)])


def test_registry_host_validate_lists_host_problems():
    host = registry_host()
    tree = host.surface().with_edits(
        {
            "tool/nope/description": "x",
            "tool/search/description": "  ",
            "tool/fetch/description": None,
            "hook/ghost/config.json": "{}",
            "hook/budget/config.json": '{"max_usd": 1, "bogus": 2}',
        }
    )
    assert host.validate(tree) == [
        "hook/budget/config.json: unknown BudgetHook config keys: ['bogus']",
        "hook/ghost/config.json: no tunable hook named 'ghost'",
        "tool/fetch/description: a tool description cannot be deleted",
        "tool/nope/description: no tool named 'nope'",
        "tool/search/description: a tool description cannot be empty",
    ]


def test_apply_tree_returns_an_edited_copy():
    registry = make_registry()
    tree = HarnessTree(
        files={
            "tool/search/description": "Use for every question.",
            "tool/search/prompt_hint": "Search first.",
            "tool/fetch/description": "Fetch one document.",
            "prompt/system": "ignored here",
        }
    )
    edited = apply_tree(registry, tree)
    assert edited is not registry
    assert edited.get("search").description == "Use for every question."
    assert edited.get("search").prompt_hint == "Search first."
    assert edited.get("fetch").prompt_hint is None  # absent hint path = no hint
    assert edited.get("delete").description == "Delete a document."  # no paths: unchanged
    assert registry.get("search").description == "Search the archive."
    with pytest.raises(OptimiserError, match="no tool named 'ghost'"):
        apply_tree(registry, HarnessTree(files={"tool/ghost/description": "x"}))


async def run_case(host, tree, agent, *, case=CASE, run_id="opt_1/base/c1"):
    model = InstrumentedModel(agent.port, ledger=Ledger(), hooks=[TranscriptHook()])
    return await host.run(tree, case, run_id=run_id, model=model)


async def test_registry_host_applies_the_tree_in_memory():
    host, agent = registry_host(), LoopAgent()
    base = await run_case(host, host.surface(), agent)
    assert (base.ok, base.delivered, base.output) == (True, True, "no idea")
    assert base.metrics["tool_calls"] == 0
    tree = host.surface().with_edits(
        {
            "prompt/system": "Always search before you answer.",
            "tool/search/prompt_hint": "One query per question.",
        }
    )
    run = await run_case(host, tree, agent)
    assert run.output == "ANSWER: Section 80C covers rent relief"
    assert run.metrics["tool_calls"] == 1
    assert run.metrics["tool_calls.search"] == 1
    assert run.metrics["steps"] == 2
    assert run.metrics["distinct_tools"] == 1
    system = system_of(agent.requests[-1])
    assert system.startswith("Always search before you answer.\n\nBe brief.")
    assert "- search: One query per question." in system
    assert "- fetch: Use after search." in system
    assert '[ACTION] search {"query": "rent relief", "token": "***"}' in run.transcript
    assert "[OBSERVATION] Section 80C covers rent relief" in run.transcript
    assert host.registry.get("search").prompt_hint is None  # nothing applied to the host


async def test_registry_host_withholds_confirm_tool_arguments_and_reports_pending():
    host, agent = registry_host(), LoopAgent()
    tree = host.surface().with_edits({"prompt/system": "DELETE things"})
    run = await run_case(host, tree, agent)
    assert "sk-secret" not in run.transcript
    assert "[ACTION] delete (arguments withheld)" in run.transcript
    assert run.metrics["tool_errors"] == 1  # no approval port: denied
    assert run.ok and run.delivered


async def test_registry_host_turns_errors_into_failed_runs():
    host, agent = registry_host(), LoopAgent()
    tree = host.surface().with_edits({"prompt/system": "BROKEN"})
    run = await run_case(host, tree, agent)
    assert (run.ok, run.delivered) == (False, False)
    assert run.error == "internal-error: RuntimeError"
    assert run.case_id == "c1"


async def test_registry_host_lets_cancellation_through():
    async def hang(request):
        await asyncio.sleep(10)

    from agent_adapter.models.callable import CallableModel

    host = registry_host()
    task = asyncio.ensure_future(
        host.run(host.surface(), CASE, run_id="r", model=CallableModel(hang))
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_registry_host_applies_hook_config_and_reports_hook_activity():
    ledger = Ledger()
    host = registry_host(hooks=[BudgetHook(5.0)], executor=Executor(ledger=ledger))
    tree = host.surface().with_edits(
        {"prompt/system": "always search", "hook/budget/config.json": '{"max_usd": 0}'}
    )
    model = InstrumentedModel(LoopAgent().port, ledger=ledger)
    run = await host.run(tree, CASE, run_id="opt_1/x/c1", model=model)
    # The tuned budget (max_usd 0) stops the tool call; the host's own hook is untouched.
    assert host.hooks[0].max_usd == 5.0
    assert run.metrics["hook_fired.budget"] == 1
    assert run.metrics["tool_errors"] == 1
    assert "budget-exceeded" in run.transcript


class Q(BaseModel):
    question: str
    answer: str = ""


async def answer(state: Q, ctx: ToolContext) -> dict:
    model = node_model(ctx)
    response = await model.generate(
        LLMRequest(
            messages=(
                Message(role="system", content=node_prompt(ctx) or ""),
                Message(role="user", content=state.question),
            ),
            stage="answer",
        )
    )
    return {"answer": response.text}


def qa_flow() -> Flow[Q]:
    flow = Flow("qa", state=Q)
    flow.node("answer", answer, prompt="Answer briefly.")
    flow.node("tidy", lambda s: {"answer": s.answer.strip()})
    return flow.edge("answer", "tidy").end("tidy")


class EchoAgent:
    def __init__(self):
        from agent_adapter.models.callable import CallableModel

        self.port = CallableModel(self._answer, provider="openai", model="gpt-4.1")

    async def _answer(self, request):
        if "FAIL" in system_of(request):
            raise RuntimeError("down")
        cite = "cite" in system_of(request)
        return reply(f" {user_of(request)}: {'80C' if cite else '?'} ")


def flow_host(**kwargs) -> FlowHost:
    return FlowHost(
        qa_flow(),
        initial_state=lambda case: Q(question=case.input),
        output=lambda state: state.answer,
        cases=[CASE],
        domain="tax",
        **kwargs,
    )


async def test_flow_host_surface_and_validation():
    host = flow_host(hooks=[BudgetHook(1.0)], allow_safety_tuning=True)
    assert host.surface().files == {
        "flow/qa/answer/prompt": "Answer briefly.",
        "hook/budget/config.json": json.dumps(
            {"max_usd": 1.0, "on_exceed": "stop", "scope": "run"}, indent=2, sort_keys=True
        ),
    }
    assert host.caps().kinds == {"flow", "hook"}
    tree = host.surface().with_edits(
        {
            "flow/qa/answer/prompt": None,
            "flow/qa/tidy/prompt": "x",
            "flow/other/answer/prompt": "x",
        }
    )
    assert host.validate(tree) == [
        "flow/other/answer/prompt: no flow named 'other'",
        "flow/qa/answer/prompt: a node prompt cannot be deleted",
        "flow/qa/tidy/prompt: node 'tidy' has no prompt to edit",
    ]


async def test_flow_host_runs_with_node_prompts_from_the_tree():
    host, agent = flow_host(), EchoAgent()
    base = await run_case(host, host.surface(), agent)
    assert (base.ok, base.delivered, base.output) == (True, True, "rent relief: ?")
    run = await run_case(host, host.surface().with_edits({"flow/qa/answer/prompt": "cite"}), agent)
    assert run.output == "rent relief: 80C"
    assert "[NODE] answer" in run.transcript
    assert "[ASSISTANT] answer:  rent relief: 80C" in run.transcript
    assert run.metrics["nodes"] == 2
    assert run.metrics["llm_calls"] == 1
    assert host.flow.prompts() == {"answer": "Answer briefly."}


async def test_flow_host_turns_node_errors_into_failed_runs():
    host = flow_host()
    tree = host.surface().with_edits({"flow/qa/answer/prompt": "FAIL"})
    run = await run_case(host, tree, EchoAgent())
    assert not run.ok and run.error.startswith("flow-error:")
    assert "[OBSERVATION ERROR] node answer" in run.transcript


class Stricter(BudgetHook):
    name = "stricter"


def test_safety_hooks_are_not_tunable_unless_the_host_allows_it():
    hooks = [
        BudgetHook(5.0),
        Stricter(1.0),
        GuardrailHook(lambda stage, payload: True),
        LoggingHook(),
        RateLimitHook({"tool": Rate(1, 1.0)}),
        BlockHook(),
    ]
    host = RegistryHost(make_registry(), layout=PromptLayout(static="s"), cases=[CASE], hooks=hooks)
    assert [p for p in host.surface().files if p.startswith("hook/")] == ["hook/block/config.json"]
    assert set(host.caps().hook_schemas) == {"block"}
    loosened = host.surface().with_edits(
        {"hook/budget/config.json": '{"max_usd": 1e9, "on_exceed": "warn"}'}
    )
    assert host.validate(loosened) == ["hook/budget/config.json: no tunable hook named 'budget'"]
    flow = flow_host(hooks=[BudgetHook(1.0)])
    assert not [p for p in flow.surface().files if p.startswith("hook/")]


async def test_a_non_safety_hook_is_tuned_and_its_activity_reported():
    host = registry_host(hooks=[BlockHook()], allow_safety_tuning=False)
    tree = host.surface().with_edits(
        {"prompt/system": "always search", "hook/block/config.json": '{"tools": ["search"]}'}
    )
    run = await run_case(host, tree, LoopAgent())
    assert run.metrics["hook_fired.block"] == 1
    assert run.output == "ANSWER: blocked by policy"
    assert host.hooks[0].tools == ()


class ConfirmCaller:
    def __init__(self):
        from agent_adapter.models.callable import CallableModel

        self.port = CallableModel(self._answer, provider="openai", model="gpt-4.1")

    async def _answer(self, request):
        from tests.optimiser.fakes import tool_reply

        return tool_reply("delete", {"doc_id": "d1", "api_key": "sk-secret"})


async def test_flow_host_withholds_confirm_tool_arguments():
    async def node(state, ctx):
        await node_model(ctx).generate(
            LLMRequest(messages=(Message(role="user", content=state.question),), stage="answer")
        )
        return {"answer": "done"}

    flow = Flow("qa2", state=Q)
    flow.node("answer", node, prompt="p")
    flow.end("answer")
    host = FlowHost(
        flow,
        initial_state=lambda case: Q(question=case.input),
        output=lambda state: state.answer,
        cases=[CASE],
        executor=Executor(make_registry()),
    )
    run = await run_case(host, host.surface(), ConfirmCaller())
    assert "sk-secret" not in run.transcript
    assert "[ACTION] delete (arguments withheld)" in run.transcript


async def test_flow_host_never_raises_when_listing_confirm_tools_fails(monkeypatch):
    host = flow_host()

    def broken(executor):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(host, "_confirm_tools", broken)
    run = await run_case(host, host.surface(), EchoAgent())
    assert not run.ok and run.error == "internal-error: RuntimeError"
