"""Prices, a request and builders shared by the InstrumentedModel tests."""

from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.types import Usage
from agent_adapter.cost import Ledger, Prices
from agent_adapter.models import InstrumentedModel
from agent_adapter.tracing import EventLog
from agent_adapter.tracing.events import REQUIRED_DATA
from tests.fakes.models import ScriptedModel, text

PRICES = Prices.load(
    {"version": "t1", "models": {"scripted": {"provider": "openai", "input": 1.0, "output": 2.0}}}
)
USAGE = Usage(input_tokens=1000, output_tokens=500, cache_write_by_ttl={"5m": 0})
COST = (1000 * 1.0 + 500 * 2.0) / 1e6
HI = LLMRequest(messages=(Message(role="user", content="hi"),), stage="answer")


def scripted(*replies):
    return ScriptedModel(list(replies) or [text("ok", usage=USAGE)])


def instrumented(port, **kwargs):
    kwargs.setdefault("ledger", Ledger(prices=PRICES))
    kwargs.setdefault("sink", EventLog())
    return InstrumentedModel(port, **kwargs)


def check_required(events):
    for event in events:
        assert REQUIRED_DATA[event.type] <= set(event.data), event.type
