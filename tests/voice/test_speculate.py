import asyncio

import pytest

from agent_adapter import CallableModel, MemoryCache, Registry
from agent_adapter.errors import ToolUsageError
from agent_adapter.exec import Executor
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.speculate import SpeculativeTools
from agent_adapter.voice.speed import Speed
from agent_adapter.voice.testing import RecordingSpeaker, settle
from agent_adapter.voice.timing import TurnTiming
from tests.fakes.models import ScriptedModel, calls, text
from tests.fakes.sinks import RecordingLedger, RecordingSink

ran: list[str] = []
registry = Registry()


@registry.tool(effects="read_only", cache_ttl=60)
async def lookup(city: str) -> str:
    """Weather for a city."""
    ran.append(f"lookup {city}")
    await asyncio.sleep(0.05)
    return f"{city}: sunny"


@registry.tool(effects="read_only")
async def unlisted(city: str) -> str:
    """Read-only but not opted in."""
    ran.append(f"unlisted {city}")
    return "ok"


@registry.tool(effects="destructive")
async def book(show: str) -> str:
    """Book a show."""
    ran.append(f"book {show}")
    return f"booked {show}"


@registry.tool(effects="read_only", confirm=True)
async def peek(city: str) -> str:
    """Read-only, but asks first."""
    return "peeked"


def make(model, *, tools=("lookup",), sink=None, ledger=None, **speed):
    ran.clear()
    executor = Executor(registry, sink=sink, ledger=ledger, cache=MemoryCache())
    spec_tools = SpeculativeTools(tools) if tools else None
    speaker = RecordingSpeaker()
    agent = VoiceAgent(model, executor, speaker, speed=Speed(tools=spec_tools, **speed))
    return agent, speaker


async def test_a_speculative_reply_is_held_then_kept():
    model = ScriptedModel([text("It is sunny in Oslo.")])
    agent, speaker = make(model)
    agent.speculate("weather in oslo")
    await settle()
    assert speaker.started == []  # nothing plays before the turn is final
    timing = TurnTiming(started=0.0)
    await agent.user_said("Weather in Oslo?", timing=timing)
    await agent.wait()
    assert speaker.said == ["It is sunny in Oslo."]
    assert len(model.requests) == 1 and timing.adopted
    assert [m.role for m in agent.history] == ["user", "assistant"]
    assert agent.history[0].content == "weather in oslo"
    await agent.close()


async def test_other_words_drop_the_speculation():
    model = ScriptedModel([text("Oslo is sunny."), text("Rome is warm.")])
    agent, speaker = make(model)
    agent.speculate("weather in oslo")
    await settle()
    await agent.user_said("weather in rome")
    await agent.wait()
    assert speaker.said == ["Rome is warm."]
    assert agent.history[0].content == "weather in rome" and len(agent.history) == 2
    await agent.close()


async def test_a_dropped_speculation_says_nothing_and_changes_no_history():
    gate = asyncio.Event()

    async def slow(request):
        await gate.wait()
        return "never heard"

    agent, speaker = make(CallableModel(slow))
    agent.speculate("weather in oslo")
    await settle()
    agent.drop_speculation()
    gate.set()
    await settle()
    assert speaker.started == [] and agent.history == []
    await agent.close()


async def test_only_opted_in_read_only_tools_run_early():
    model = ScriptedModel(
        [calls(("lookup", {"city": "Oslo"}), ("unlisted", {"city": "Oslo"})), text("Done.")]
    )
    agent, _ = make(model)
    agent.speculate("check oslo")
    await asyncio.sleep(0.1)
    assert ran == ["lookup Oslo"]  # the unlisted one waits for the final transcript
    await agent.user_said("check oslo")
    await agent.wait()
    assert sorted(ran) == ["lookup Oslo", "unlisted Oslo"]
    await agent.close()


async def test_a_side_effect_never_runs_from_a_dropped_speculation():
    model = ScriptedModel([calls(("book", {"show": "jazz"})), text("Booked.")])
    agent, speaker = make(model)
    agent.speculate("book jazz")
    await asyncio.sleep(0.05)
    agent.drop_speculation()
    await asyncio.sleep(0.05)
    assert ran == [] and speaker.started == []
    await agent.close()


def test_opting_in_a_tool_that_changes_things_is_refused():
    for name in ("book", "peek", "missing"):
        with pytest.raises(ToolUsageError):
            make(ScriptedModel([]), tools=(name,))


async def test_the_gate_checks_the_tool_on_every_call():
    model = ScriptedModel([calls(("book", {"show": "jazz"})), text("Booked.")])
    agent, _ = make(model)
    assert agent.speed is not None and agent.speed.tools is not None
    agent.speed.tools.names = frozenset({"lookup", "book"})  # changed after the check
    agent.speculate("book jazz")
    await asyncio.sleep(0.05)
    assert ran == []
    await agent.close()


async def test_early_calls_have_a_budget_line_a_trace_tag_and_a_limit():
    sink, ledger = RecordingSink(), RecordingLedger()
    lookups = [("lookup", {"city": c}) for c in ("Oslo", "Rome")]
    model = ScriptedModel([calls(*lookups), text("Both sunny.")])
    ran.clear()
    executor = Executor(registry, sink=sink, ledger=ledger, cache=MemoryCache())
    tools = SpeculativeTools(["lookup"], max_per_turn=1)
    agent = VoiceAgent(model, executor, RecordingSpeaker(), speed=Speed(tools=tools))
    agent.speculate("oslo and rome")
    await asyncio.sleep(0.1)
    assert ran == ["lookup Oslo"]  # one early call per speculation
    assert [e.data["tool"] for e in sink.of_type("custom.voice.speculative_tool")] == ["lookup"]
    stages = [e.stage for e in ledger.entries()]
    assert stages.count("voxcat:speculative_tool") == 1
    assert "voice_speculation" in stages  # the speculative model call has its own stage
    await agent.user_said("oslo and rome")
    await agent.wait()
    assert sorted(ran) == ["lookup Oslo", "lookup Rome"]
    outcomes = [e.data["outcome"] for e in sink.of_type("custom.voice.speculation")]
    assert outcomes == ["started", "adopted"]
    await agent.close()


async def test_a_dropped_speculations_tool_call_is_shared_with_the_next_turn():
    model = ScriptedModel(
        [
            calls(("lookup", {"city": "Oslo"})),
            calls(("lookup", {"city": "Oslo"})),
            text("Sunny."),
        ]
    )
    agent, speaker = make(model)
    agent.speculate("weather oslo")
    await settle()
    assert ran == ["lookup Oslo"]  # the early lookup is running
    agent.drop_speculation()
    await agent.user_said("weather oslo please")
    await agent.wait()
    assert ran == ["lookup Oslo"]  # one real call: in flight, then from the cache
    assert speaker.said == ["Sunny."]
    await agent.close()


async def test_no_speculation_while_a_turn_runs():
    gate = asyncio.Event()

    async def held(request):
        await gate.wait()
        return "Hello."

    agent, _ = make(CallableModel(held))
    await agent.user_said("hi there")
    agent.speculate("something else")
    assert agent._fast is not None and agent._fast.speculation is None
    gate.set()
    await agent.wait()
    await agent.close()


async def test_a_dropped_speculation_stops_before_its_next_model_call():
    model = ScriptedModel([calls(("lookup", {"city": "Oslo"})), text("never asked")])
    agent, _ = make(model)
    agent.speculate("weather oslo")
    await settle()
    assert agent._fast is not None and agent._fast.speculation is not None
    task = agent._fast.speculation.task
    agent.drop_speculation()
    assert task is not None
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
    assert task.cancelled() and len(model.requests) == 1  # the lookup landed; no second call
    await agent.close()


async def test_a_failed_speculation_falls_back_to_a_fresh_turn():
    model = ScriptedModel([RuntimeError("provider hiccup"), text("Hello again.")])
    agent, speaker = make(model)
    agent.speculate("hello there")
    await settle()
    await agent.user_said("hello there")
    await agent.wait()
    assert speaker.said == ["Hello again."] and len(model.requests) == 2
    await agent.close()
