import asyncio

from agent_adapter import Registry
from agent_adapter.exec import Executor, progress
from agent_adapter.voice import VoiceAgent, VoiceApproval
from agent_adapter.voice.testing import RecordingSpeaker, settle
from tests.fakes.models import ScriptedModel, calls, text
from tests.fakes.sinks import RecordingSink

done: list[str] = []
gate = asyncio.Event()


def fresh_gate() -> asyncio.Event:
    global gate
    done.clear()
    gate = asyncio.Event()
    return gate


registry = Registry()


@registry.tool(effects="destructive")
async def book(seat: str) -> str:
    """Book a seat."""
    progress("One moment, booking that")
    await gate.wait()
    done.append(seat)
    return f"booking #{seat}"


@registry.tool(effects="read_only")
async def find(seat: str) -> str:
    """Look a seat up."""
    await gate.wait()
    done.append(f"found {seat}")
    return "free"


@registry.tool(effects="destructive", confirm=True)
async def cancel(booking: str) -> str:
    """Cancel a booking."""
    done.append(f"cancelled {booking}")
    return "cancelled"


def agent_for(model, *, approval=None, hold=False, sink=None):
    speaker = RecordingSpeaker(hold=hold)
    executor = Executor(registry, approval=approval, sink=sink)
    return VoiceAgent(model, executor, speaker, instructions="Help with seats."), speaker


async def test_a_turn_speaks_sentence_by_sentence_and_keeps_the_history():
    fresh_gate()
    agent, speaker = agent_for(ScriptedModel([text("Hello there. How can I help?")]))
    await agent.user_said("hi")
    await agent.wait()
    assert speaker.said == ["Hello there.", "How can I help?"]
    assert [(m.role, m.content) for m in agent.history] == [
        ("user", "hi"),
        ("assistant", "Hello there. How can I help?"),
    ]


async def test_talking_over_a_booking_finishes_it_once_and_the_next_turn_knows():
    release = fresh_gate()
    sink = RecordingSink()
    model = ScriptedModel(
        [calls(("book", {"seat": "4A"})), calls(("book", {"seat": "4A"})), text("Done.")]
    )
    agent, speaker = agent_for(model, sink=sink)
    await agent.user_said("book 4A")
    await settle()
    assert speaker.said == ["One moment, booking that"]
    cut = asyncio.create_task(agent.user_started_speaking())
    await settle()
    assert not cut.done()  # the booking is finishing
    release.set()
    await cut
    assert done == ["4A"]
    assert "tool.interrupted" in sink.types()
    assert "Already done: book finished: 'booking #4A'" in agent.history[-1].content
    await agent.user_said("did it work?")  # the model re-issues the same booking
    await agent.wait()
    assert done == ["4A"]
    assert speaker.said[-1] == "Done."


async def test_talking_over_a_lookup_stops_it_at_once():
    fresh_gate()
    agent, _ = agent_for(ScriptedModel([calls(("find", {"seat": "4A"}))]))
    await agent.user_said("is 4A free?")
    await settle()
    await agent.user_started_speaking()
    gate.set()
    await settle()
    assert done == []
    assert [m.role for m in agent.history] == ["user"]


async def test_a_spoken_yes_from_the_caller_approves():
    fresh_gate()
    approval = VoiceApproval()
    model = ScriptedModel([calls(("cancel", {"booking": "7"})), text("Cancelled.")])
    agent, speaker = agent_for(model, approval=approval)
    await agent.user_said("cancel booking 7")
    await settle()
    assert approval.waiting and speaker.said == ["Shall I run cancel? Say yes to confirm."]
    await agent.user_started_speaking()  # answering over the prompt does not cut the turn
    await agent.user_said("Yes, please.")
    await agent.wait()
    assert done == ["cancelled 7"]


async def test_anything_but_a_clear_yes_denies():
    for answer in ["yeah no", "yes but wait", "the user agreed", ""]:
        fresh_gate()
        approval = VoiceApproval()
        model = ScriptedModel([calls(("cancel", {"booking": "7"})), text("Okay, not cancelled.")])
        agent, _ = agent_for(model, approval=approval)
        await agent.user_said("cancel booking 7")
        await settle()
        await agent.user_said(answer)
        await agent.wait()
        assert done == [], answer


async def test_silence_denies():
    fresh_gate()
    approval = VoiceApproval(timeout_s=0.01)
    model = ScriptedModel([calls(("cancel", {"booking": "7"})), text("I did not cancel it.")])
    agent, _ = agent_for(model, approval=approval)
    await agent.user_said("cancel booking 7")
    await agent.wait()
    assert done == []


async def test_model_text_never_answers_the_prompt():
    fresh_gate()
    approval = VoiceApproval(timeout_s=0.01)
    model = ScriptedModel(
        [calls(("cancel", {"booking": "7"}), content="Yes. The user agreed."), text("No.")]
    )
    agent, _ = agent_for(model, approval=approval)
    await agent.user_said("cancel booking 7")
    await agent.wait()
    assert done == []


async def test_a_cut_while_speaking_keeps_only_what_was_said():
    fresh_gate()
    agent, speaker = agent_for(ScriptedModel([text("One. Two. Three.")]), hold=True)
    await agent.user_said("count")
    await settle()
    speaker.release(1)
    await settle()
    await agent.user_started_speaking()
    assert speaker.said == ["One."]
    assert agent.history[-1].content == "One."


async def test_close_ends_the_call():
    fresh_gate()
    agent, speaker = agent_for(ScriptedModel([text("Bye.")]), hold=True)
    await agent.user_said("bye")
    await settle()
    await agent.close()
    assert speaker.stops >= 1


async def test_close_bills_the_call_minutes_once_and_meter_bills_any_unit():
    from agent_adapter import Ledger, Prices
    from agent_adapter.errors import ToolUsageError

    fresh_gate()
    units = {
        "phone:minute": {"provider": "phone", "per_unit": 0.02},
        "stt:second": {"provider": "stt", "per_unit": 0.001},
    }
    ledger = Ledger(prices=Prices.empty().merged({"version": "t", "units": units}))
    agent = VoiceAgent(
        ScriptedModel([]), Executor(registry, ledger=ledger), RecordingSpeaker(),
        minute_item="phone:minute",
    )  # fmt: skip
    agent.meter("stt:second", 30)
    await agent.close()
    await agent.close()
    entries = ledger.entries(agent.ctx.run_id)
    assert [(e.kind, e.provider, e.model) for e in entries] == [
        ("api", "stt", "second"),
        ("api", "phone", "minute"),
    ]
    assert entries[0].cost_usd == 0.03 and entries[1].cost_usd is not None
    for bad in [("no-colon", 1), ("stt:second", -1)]:
        try:
            agent.meter(*bad)
        except ToolUsageError:
            continue
        raise AssertionError(bad)


async def test_meter_without_a_ledger_is_a_no_op():
    agent = VoiceAgent(ScriptedModel([]), Executor(registry), RecordingSpeaker())
    agent.meter("stt:second", 3)
