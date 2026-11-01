"""A failing tool's exception text never reaches INFO+ logs (it can carry secrets)."""

import logging

import pytest
from pydantic import BaseModel

from agent_adapter import Registry, ToolCall
from agent_adapter.approval import CallbackApproval
from agent_adapter.core.errors import FlowError, log_failure
from agent_adapter.exec import Executor, Step
from agent_adapter.flow import Flow, FlowRunner
from agent_adapter.hooks.base import Hook
from agent_adapter.tracing.events import Event, emit

registry = Registry()


@registry.tool(retries=1)
async def connect(host: str) -> str:
    """Connect to a host."""
    raise RuntimeError(f"login to {host} failed: password=SEKRIT1")


async def test_a_secret_in_the_exception_text_is_never_logged_at_info_or_above(caplog):
    with caplog.at_level(logging.DEBUG, logger="agent_adapter"):
        result = await Executor(registry).run_call(
            ToolCall(id="c1", name="connect", arguments={"host": "db"})
        )
    assert result.status == "error"
    loud = [r for r in caplog.records if r.levelno >= logging.INFO]
    errors = [r for r in loud if r.levelno == logging.ERROR]
    assert len(errors) == 1
    line = errors[0].getMessage()
    assert "connect" in line and "RuntimeError" in line and "c1" in line
    for record in loud:
        assert record.exc_info is None
        assert "SEKRIT1" not in record.getMessage()
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG and r.exc_info]
    assert debug, "the full traceback stays available at DEBUG"


SECRET = "password=SEKRIT2"


def _assert_quiet(caplog, *, level: int, needles: tuple[str, ...]) -> None:
    """A record at `level` names `needles`; no INFO+ record holds the secret or a traceback."""
    loud = [r for r in caplog.records if r.levelno >= logging.INFO]
    lines = [r.getMessage() for r in loud if r.levelno == level]
    assert any(all(needle in line for needle in needles) for line in lines), lines
    for record in loud:
        assert record.exc_info is None
        assert SECRET not in record.getMessage()
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG and r.exc_info]
    assert debug and SECRET in str(debug[0].exc_info[1]), "the traceback stays at DEBUG"


@registry.tool
async def echo(text: str) -> str:
    """Echo the text."""
    return text


async def test_a_failing_plan_args_callable_logs_only_its_type_at_warning(caplog):
    def args(deps):
        raise ValueError(f"cannot build args: {SECRET}")

    steps = [Step(id="build", tool="echo", args=args)]
    with caplog.at_level(logging.DEBUG, logger="agent_adapter"):
        results = await Executor(registry).run_plan(steps)
    assert results["build"].status == "error"
    _assert_quiet(caplog, level=logging.WARNING, needles=("build", "echo", "ValueError"))


def _broken_rule(call, ctx):
    raise LookupError(f"no policy for {call.arguments['text']}")


@registry.tool(confirm=_broken_rule)
async def guarded(text: str) -> str:
    """A tool whose confirm rule fails."""
    return text


async def test_a_failing_confirm_rule_logs_only_its_type_at_error(caplog):
    with caplog.at_level(logging.DEBUG, logger="agent_adapter"):
        result = await Executor(registry).run_call(
            ToolCall(id="c2", name="guarded", arguments={"text": SECRET})
        )
    assert result.status == "error"
    _assert_quiet(caplog, level=logging.ERROR, needles=("guarded", "c2", "LookupError"))


def test_a_failing_event_sink_logs_only_its_type_at_warning(caplog):
    class Broken:
        def emit(self, event):
            raise RuntimeError(f"sink down: {SECRET}")

    with caplog.at_level(logging.DEBUG, logger="agent_adapter"):
        emit(Broken(), Event("custom.ping", run_id="r1"))
    _assert_quiet(caplog, level=logging.WARNING, needles=("custom.ping", "RuntimeError"))


def test_log_failure_names_the_caller_and_keeps_the_text_at_debug(caplog):
    log = logging.getLogger("agent_adapter.test_hygiene")
    with caplog.at_level(logging.DEBUG, logger="agent_adapter"):
        log_failure(log, logging.WARNING, KeyError(SECRET), "lookup %s failed", "k1")
    warning, debug = caplog.records
    assert warning.getMessage() == "lookup k1 failed (KeyError)"
    assert (
        warning.funcName
        == debug.funcName
        == ("test_log_failure_names_the_caller_and_keeps_the_text_at_debug")
    )
    assert debug.levelno == logging.DEBUG and debug.exc_info is not None


async def test_a_failing_approval_callback_logs_only_its_type_at_error(caplog):
    def approver(request):
        raise RuntimeError(f"approver down: {SECRET}")

    executor = Executor(registry, approval=CallbackApproval(approver))
    with caplog.at_level(logging.DEBUG, logger="agent_adapter"):
        result = await executor.run_call(
            ToolCall(id="c3", name="needs_ok", arguments={"text": SECRET})
        )
    assert result.status == "error"
    _assert_quiet(caplog, level=logging.ERROR, needles=("needs_ok", "c3", "RuntimeError"))


@registry.tool(confirm=True)
async def needs_ok(text: str) -> str:
    """A tool that always asks for approval."""
    return text


class LeakyHook(Hook):
    async def before_tool(self, inv):
        raise RuntimeError(f"hook saw {inv.call.arguments['text']}")


async def test_a_failing_hook_logs_only_its_type_at_error(caplog):
    executor = Executor(registry, hooks=[LeakyHook()])
    with caplog.at_level(logging.DEBUG, logger="agent_adapter"):
        result = await executor.run_call(ToolCall(id="c4", name="echo", arguments={"text": SECRET}))
    assert result.status == "error"
    _assert_quiet(caplog, level=logging.ERROR, needles=("LeakyHook", "before_tool", "RuntimeError"))


class Note(BaseModel):
    text: str = ""


async def test_a_failing_flow_node_logs_only_its_type_at_error(caplog):
    def explode(state):
        raise RuntimeError(f"node saw {SECRET}")

    flow = Flow("leaky", state=Note).node("boom", explode).end("boom")
    with caplog.at_level(logging.DEBUG, logger="agent_adapter"), pytest.raises(FlowError):
        await FlowRunner(Executor(registry)).run(flow, {"text": SECRET})
    _assert_quiet(caplog, level=logging.ERROR, needles=("leaky", "boom", "RuntimeError"))
