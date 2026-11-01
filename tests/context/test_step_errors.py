"""on_step_error: a failing step is skipped (logged, an event) and the main call goes on."""

import logging

import pytest

from agent_adapter.context import (
    CompressingModel,
    CompressionPolicy,
    MyOwnStep,
    RemoveOldMessages,
    Summarize,
    compress,
)
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, ModelUnavailable
from agent_adapter.models.callable import CallableModel
from agent_adapter.runtime.current import use_context
from tests.context.test_wiring_model import HISTORY, PAD, _overflow, _request
from tests.fakes.models import ScriptedModel, text
from tests.fakes.sinks import RecordingSink


async def _down(request):
    raise ConnectionError("summariser at 10.0.0.7 is down")


DOWN = CallableModel(_down, provider="fake", model="summariser")


def _policy(**extra):
    steps = extra.pop("steps", None) or [
        Summarize(DOWN, keep_recent_messages=1),
        RemoveOldMessages(keep_first=0, keep_last=3),
    ]
    return CompressionPolicy(start_when_tokens_over=150, shrink_to_tokens=120, steps=steps, **extra)


async def test_a_failing_step_is_skipped_and_the_next_one_runs(caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.context.compress"):
        smaller, report = await compress(list(HISTORY), _policy())
    assert report.steps_used == ("remove_old_messages",) and len(smaller) < len(HISTORY)
    (failure,) = report.steps_failed
    assert failure.step == "summarize" and failure.latency_ms >= 0
    assert "compression step summarize failed; skipped" in caplog.text
    assert "10.0.0.7" not in caplog.text  # exception text goes to DEBUG only


async def test_on_step_error_raise_fails_fast():
    with pytest.raises(ConnectionError):
        await compress(list(HISTORY), _policy(on_step_error="raise"))


async def test_a_config_error_is_never_skipped():
    def wrong(messages):
        return "not a list"

    policy = _policy(steps=[MyOwnStep(wrong)])
    with pytest.raises(ConfigError, match="must return a list"):
        await compress(list(HISTORY), policy)


def test_on_step_error_takes_skip_or_raise_only():
    assert CompressionPolicy(model_max_tokens=100).on_step_error == "skip"
    with pytest.raises(ConfigError, match="on_step_error"):
        CompressionPolicy(model_max_tokens=100, on_step_error="ignore")


async def test_the_main_call_proceeds_and_compress_failed_is_emitted():
    sink = RecordingSink()
    model = ScriptedModel([text("answer")])
    with use_context(ToolContext(run_id="run_1", sink=sink)):
        reply = await model.generate(_request(compression=_policy()))
    assert reply.text == "answer" and len(model.requests[0].messages) < len(HISTORY)
    failed, done = sink.of_type("context.")
    assert (failed.type, failed.data["step"]) == ("context.compress_failed", "summarize")
    assert failed.data["error_kind"] and failed.data["trigger"] == "threshold"
    assert done.type == "context.compressed" and done.data["steps"] == ["remove_old_messages"]
    assert PAD[:20] not in str(failed.data)


async def test_a_failing_step_in_the_stronger_policy_still_retries_once():
    async def boom(messages):
        raise ModelUnavailable("down", provider="x")

    stronger = CompressionPolicy(
        start_when_tokens_over=100,
        shrink_to_tokens=50,
        steps=[MyOwnStep(boom), RemoveOldMessages(keep_first=0, keep_last=1)],
    )
    inner = ScriptedModel([_overflow(), text("ok")])
    policy = _policy(steps=[MyOwnStep(boom)], if_still_too_long=stronger)
    reply = await CompressingModel(inner, policy).generate(_request())
    assert reply.text == "ok" and len(inner.requests) == 2
    assert len(inner.requests[1].messages) < len(inner.requests[0].messages)
