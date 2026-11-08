import functools

import pytest
from pydantic import BaseModel

from agent_adapter import chat as chat_contract
from agent_adapter import viewer
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.cost.ledger import Ledger
from agent_adapter.cost.prices import Prices
from agent_adapter.evals import (
    CallableTarget,
    ChatTarget,
    FlowTarget,
    ToolLoopTarget,
    as_target,
    run_eval,
)
from agent_adapter.evals import targets as targets_module
from agent_adapter.evals.graders import contains, exact, tools_called
from agent_adapter.exec.executor import Executor
from agent_adapter.flow import Call, Flow
from agent_adapter.viewer.chat import contract as viewer_contract
from tests.evals.fakes import case, shop_registry
from tests.fakes.approval import ScriptedApproval
from tests.fakes.models import ScriptedModel, calls, text
from tests.fakes.sinks import RecordingSink

PRICES = Prices.empty().merged(
    {"version": "t", "models": {"scripted": {"provider": "scripted", "input": 1.0, "output": 1.0}}}
)


class Ticket(BaseModel):
    text: str
    reply: str = ""


def test_the_chat_page_and_evals_share_one_chat_contract():
    for name in ("Attachment", "ChatAgent", "ChatMessage", "ChatReply", "ChatSession"):
        shared = getattr(chat_contract, name)
        assert getattr(viewer_contract, name) is shared
        assert getattr(viewer, name, shared) is shared  # the viewer re-exports the same class
    assert targets_module.ChatSession is chat_contract.ChatSession
    assert targets_module.ChatMessage is chat_contract.ChatMessage


def plain(question):
    return f"echo {question}"


async def chat(message, session):
    return f"chat {message.text}"


def with_ctx(question, ctx):
    return ctx.run_id


class Custom:
    name = "custom"

    async def run_case(self, case, *, executor, ctx):
        return "custom"


def test_as_target_picks_by_arity():
    assert isinstance(as_target(plain), CallableTarget)
    assert isinstance(as_target(chat), ChatTarget)
    assert isinstance(as_target(with_ctx), CallableTarget)
    assert isinstance(as_target(functools.partial(chat, session=None)), CallableTarget)
    custom = Custom()
    assert as_target(custom) is custom
    assert as_target(plain).name == "plain"


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (lambda: 1, "0 required arguments"),
        (lambda a, b, c: 1, "3 required arguments"),
        ("text", "cannot evaluate a str"),
        (Flow("f", state=Ticket), "wrap a Flow in FlowTarget"),
    ],
)
def test_as_target_refuses_what_it_cannot_run(target, message):
    with pytest.raises(ConfigError, match=message):
        as_target(target)


def test_targets_refuse_bad_arguments():
    with pytest.raises(ConfigError):
        CallableTarget("x")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        ChatTarget(1)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="Registry"):
        ToolLoopTarget(ScriptedModel([]), {})  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="needs a Flow"):
        FlowTarget("f", initial_state=dict)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="initial_state"):
        FlowTarget(Flow("f", state=Ticket), initial_state=None)  # type: ignore[arg-type]


async def test_callable_targets_run_sync_async_and_streamed():
    async def streamed(question):
        yield "a"
        yield "b"

    async def answer(question):
        return {"q": question}

    report = await run_eval(plain, [case()], [contains()])
    assert report.cases[0].output == "echo where is 1042"
    streamed_report = await run_eval(streamed, [case(expected="ab")], [exact()])
    assert streamed_report.cases[0].passed
    structured = await run_eval(answer, [case(expected=None)], [tools_called(forbid=["x"])])
    assert structured.cases[0].output == '{"q": "where is 1042"}'


async def test_a_callable_taking_ctx_joins_the_case_run():
    registry = shop_registry()
    executor = Executor(registry)

    async def agent(question, ctx):
        return await executor.run("order_status", {"order_id": "1042"}, ctx)

    report = await run_eval(
        agent, [case()], [exact(), tools_called("order_status")], executor=executor
    )
    result = report.cases[0]
    assert result.passed and result.tool_calls == ("order_status",)


async def test_chat_targets_get_a_fresh_session_in_the_case_run():
    seen = {}

    def agent(message, session):
        seen["types"] = (
            isinstance(message, chat_contract.ChatMessage),
            isinstance(session, chat_contract.ChatSession),
        )
        seen.update(
            text=message.text,
            run=session.run_id,
            ctx=session.context().run_id,
            history=session.history,
            state=dict(session.state),
            id=session.id,
        )
        return "shipped"

    report = await run_eval(agent, [case("c9")], [contains()])
    assert seen.pop("types") == (True, True)
    assert seen == {
        "text": "where is 1042",
        "run": report.cases[0].run_id,
        "ctx": report.cases[0].run_id,
        "history": (),
        "state": {},
        "id": "eval-c9",
    }


async def test_chat_targets_stream_and_refuse_bad_replies():
    async def streaming(message, session):
        for word in ("ship", "ped"):
            yield word

    async def bad(message, session):
        return 3

    async def bad_chunks(message, session):
        yield 3

    assert (await run_eval(streaming, [case()], [exact()])).cases[0].passed
    for agent in (bad, bad_chunks):
        result = (await run_eval(agent, [case()], [exact()])).cases[0]
        assert result.error.startswith("tool-usage-error:") and not result.passed


def loop_executor(**options):
    return Executor(shop_registry(), ledger=Ledger(prices=PRICES), **options)


async def test_tool_loop_target_runs_tools_and_prices_the_case():
    from agent_adapter.core.types import Usage

    usage = Usage(input_tokens=100, output_tokens=10)
    model = ScriptedModel(
        [calls(("order_status", {"order_id": "1042"}), usage=usage), text("shipped", usage=usage)]
    )
    target = ToolLoopTarget(model, shop_registry(), system="Answer order questions.")
    report = await run_eval(target, [case()], [exact()], executor=loop_executor())
    result = report.cases[0]
    assert result.passed and result.tool_calls == ("order_status",)
    assert result.cost_usd == pytest.approx(220 / 1_000_000)
    assert model.requests[0].messages[0].role == "system"


async def test_tool_loop_parsed_output_and_unfinished_loops():
    from agent_adapter.core.messages import OutputSchema

    schema = OutputSchema(name="o", json_schema={"type": "object", "properties": {}})
    model = ScriptedModel([text('{"status": "shipped"}', parsed={"status": "shipped"})])
    target = ToolLoopTarget(model, shop_registry(), output_schema=schema)
    done = await run_eval(target, [case(expected=None)], [tools_called(forbid=["refund"])])
    assert done.cases[0].output == '{"status": "shipped"}'
    looping = ToolLoopTarget(
        ScriptedModel([calls(("order_status", {"order_id": "1"}))]), shop_registry(), max_steps=1
    )
    stuck = (await run_eval(looping, [case()], [exact()])).cases[0]
    assert stuck.error == "unfinished: tool loop max_steps" and not stuck.passed
    assert stuck.grades == ()


async def test_a_pending_approval_is_a_case_error_never_a_pass():
    from agent_adapter.core.errors import ApprovalPending

    model = ScriptedModel([calls(("refund", {"order_id": "1042"})), text("refunded")])
    approval = ScriptedApproval([ApprovalPending("ap1", tool="refund")])
    executor = loop_executor(approval=approval)
    result = (
        await run_eval(
            ToolLoopTarget(model, shop_registry()), [case()], [contains()], executor=executor
        )
    ).cases[0]
    assert result.error.startswith("approval-pending:") and not result.passed


async def test_confirm_tools_are_denied_without_an_approval_port():
    model = ScriptedModel([calls(("refund", {"order_id": "1042"})), text("could not refund")])
    sink = RecordingSink()
    executor = loop_executor(sink=sink)
    result = (
        await run_eval(
            ToolLoopTarget(model, shop_registry()),
            [case(expected="could not")],
            [contains()],
            executor=executor,
        )
    ).cases[0]
    assert result.passed
    assert not [e for e in sink.events if e.type == "tool.finished"]


async def test_an_approved_confirm_tool_runs():
    model = ScriptedModel([calls(("refund", {"order_id": "1042"})), text("refunded 1042")])
    approval = ScriptedApproval([ApprovalDecision(approved=True)])
    result = (
        await run_eval(
            ToolLoopTarget(model, shop_registry()),
            [case(expected="refunded")],
            [contains(), tools_called("refund")],
            executor=loop_executor(approval=approval),
        )
    ).cases[0]
    assert result.passed


async def test_flow_target_runs_the_flow_and_reads_its_output():
    registry = shop_registry()
    flow = (
        Flow("support", state=Ticket)
        .node("look", Call("order_status", args=lambda t: {"order_id": "1042"}, into="reply"))
        .end("look")
    )
    target = FlowTarget(flow, initial_state=lambda c: {"text": c.input}, output=lambda s: s.reply)
    assert target.name == "support"
    report = await run_eval(
        target, [case()], [exact(), tools_called("order_status")], executor=Executor(registry)
    )
    assert report.cases[0].passed and report.cases[0].output == "shipped"
    whole = FlowTarget(flow, initial_state=lambda c: Ticket(text="x"))
    state = (
        await run_eval(whole, [case(expected=None)], [contains()], executor=Executor(registry))
    ).cases[0]
    assert '"reply":"shipped"' in state.output


async def test_a_paused_flow_is_unfinished():
    registry = shop_registry()
    flow = (
        Flow("refunds", state=Ticket)
        .node("refund", Call("refund", args=lambda t: {"order_id": "1042"}, into="reply"))
        .end("refund")
    )
    from agent_adapter.core.errors import ApprovalPending

    approval = ScriptedApproval([ApprovalPending("ap2", tool="refund")])
    target = FlowTarget(flow, initial_state=lambda c: {"text": "x"})
    result = (
        await run_eval(
            target, [case()], [contains()], executor=Executor(registry, approval=approval)
        )
    ).cases[0]
    assert result.error == "unfinished: flow paused"


def test_a_callable_without_a_readable_signature_is_refused():
    with pytest.raises(ConfigError, match="cannot read the signature"):
        as_target(min)


async def test_an_async_iterator_without_aclose_is_joined():
    class Chunks:
        def __init__(self):
            self.left = ["ship", "ped"]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.left:
                raise StopAsyncIteration
            return self.left.pop(0)

    report = await run_eval(lambda q: Chunks(), [case()], [exact()])
    assert report.cases[0].passed


async def test_a_pending_loop_without_a_pending_result_is_still_unfinished(monkeypatch):
    from agent_adapter.core.types import Usage
    from agent_adapter.evals import targets
    from agent_adapter.exec.tool_loop import ToolLoopResult

    async def fake_loop(*args, **kwargs):
        return ToolLoopResult(
            messages=(),
            final=text("wait"),
            steps=1,
            finish="pending",
            tool_results=(),
            usage=Usage(),
        )

    monkeypatch.setattr(targets, "run_tool_loop", fake_loop)
    target = ToolLoopTarget(ScriptedModel([]), shop_registry())
    result = (await run_eval(target, [case()], [exact()])).cases[0]
    assert result.error == "unfinished: tool loop pending"


async def test_tool_call_deltas_in_a_streamed_reply_are_skipped():
    from agent_adapter.core.messages import ToolCallDelta
    from agent_adapter.evals._calls import joined

    async def reply():
        yield "ship"
        yield ToolCallDelta(index=0, name="order_status")
        yield "ped"

    assert await joined(reply(), "agent") == "shipped"
