import asyncio

import pytest

from agent_adapter import CallableModel, MemoryCache, Registry
from agent_adapter.errors import ToolUsageError
from agent_adapter.exec import Executor
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice.ack import Acknowledgements
from agent_adapter.voice.answers import AnswerCache
from agent_adapter.voice.speed import Speed
from agent_adapter.voice.testing import RecordingSpeaker, WordTTS
from agent_adapter.voice.timing import TurnTiming
from agent_adapter.voice.warm import Warmup
from tests.fakes.models import ScriptedModel, calls, text

registry = Registry()
looked: list[str] = []


@registry.tool(effects="read_only", cache_ttl=60)
async def account(caller: str) -> str:
    """The caller's account."""
    looked.append(caller)
    return f"{caller}: gold"


@registry.tool(effects="destructive")
async def charge(amount: int) -> str:
    """Charge the caller."""
    return "charged"


def agent_with(model, speaker=None, **speed):
    executor = Executor(registry, cache=MemoryCache())
    speaker = speaker or RecordingSpeaker()
    return VoiceAgent(model, executor, speaker, speed=Speed(**speed)), speaker


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_a_repeated_question_is_answered_from_the_cache():
    answers = AnswerCache()
    first, _ = agent_with(ScriptedModel([text("We open at nine.")]), answers=answers, tenant="acme")
    await first.user_said("When do you open?")
    await first.wait()
    model = ScriptedModel([])
    second, speaker = agent_with(model, answers=answers, tenant="acme")
    timing = TurnTiming(started=0.0)
    await second.user_said("when do you open", timing=timing)
    await second.wait()
    assert speaker.said == ["We open at nine."] and model.requests == []
    assert timing.answer_cached and timing.replied is not None
    assert second.history[-1].content == "We open at nine."
    assert answers.hits == 1


async def test_answers_are_per_tenant_and_expire():
    clock = Clock()
    answers = AnswerCache(ttl_s=10, clock=clock)
    answers.put("acme", "When do you open?", "At nine.")
    assert answers.get("other", "when do you open") is None
    assert answers.get("acme", "WHEN do you open!") is not None
    clock.now = 11
    assert answers.get("acme", "when do you open") is None
    with pytest.raises(ValueError):
        answers.get("", "when do you open")


async def test_short_turns_tool_replies_and_later_turns_are_not_stored():
    answers = AnswerCache()
    assert answers.put("acme", "yes", "Booked.") is None
    model = ScriptedModel(
        [calls(("account", {"caller": "ana"})), text("You are gold."), text("Nine to five.")]
    )
    agent, _ = agent_with(model, answers=answers, tenant="acme")
    await agent.user_said("what is my tier")
    await agent.wait()
    await agent.user_said("what are your hours")
    await agent.wait()
    assert len(answers) == 0  # one used a tool; the other came after the call opened


async def test_the_cached_voice_replays_stored_audio():
    answers = AnswerCache()
    tts = WordTTS(ms_per_word=10)
    voice = answers.voice(tts)
    first = [chunk async for chunk in voice.synthesize("We open at nine.")]
    answers.put("acme", "when do you open", "We open at nine.")
    again = [chunk async for chunk in voice.synthesize("We open at nine.")]
    assert again == first and tts.spoken == ["We open at nine."]


def test_answers_need_a_tenant():
    with pytest.raises(ToolUsageError):
        Speed(answers=AnswerCache())


async def test_a_slow_reply_gets_an_acknowledgement_first():
    async def slow(request):
        await asyncio.sleep(0.05)
        return "Here you go."

    acks = Acknowledgements(["Mm-hm."], fast_ms=20, expect_ms=500)
    agent, speaker = agent_with(CallableModel(slow), acks=acks)
    await agent.warm_up()
    timing = TurnTiming(started=0.0)
    await agent.user_said("tell me", timing=timing)
    await agent.wait()
    assert speaker.said == ["Mm-hm.", "Here you go."]
    assert timing.acked is not None and timing.replied is not None
    assert timing.acked < timing.replied


async def test_a_fast_reply_gets_no_acknowledgement():
    acks = Acknowledgements(["Mm-hm."], fast_ms=200, expect_ms=50)
    agent, speaker = agent_with(ScriptedModel([text("Quick.")]), acks=acks)
    await agent.warm_up()
    timing = TurnTiming(started=0.0)
    await agent.user_said("tell me", timing=timing)
    await agent.wait()
    assert speaker.said == ["Quick."] and timing.acked is None


async def test_unrendered_lines_never_play():
    async def slow(request):
        await asyncio.sleep(0.03)
        return "Here."

    acks = Acknowledgements(["Mm-hm."], fast_ms=0, expect_ms=500)
    agent, speaker = agent_with(CallableModel(slow), acks=acks)
    await agent.user_said("tell me")  # no warm-up: nothing is rendered
    await agent.wait()
    assert speaker.said == ["Here."]


def test_acknowledgements_learn_the_reply_time():
    acks = Acknowledgements(fast_ms=300, expect_ms=1000, smoothing=0.5)
    assert acks.delay_ms(0) == 0  # slow: acknowledge at once
    for _ in range(3):
        acks.learn(100)
    assert acks.expected_ms < 300 and acks.delay_ms(0) == 300


class Pool:
    def __init__(self) -> None:
        self.warmed = 0

    async def warm(self) -> None:
        self.warmed += 1


class Broken:
    async def warm(self) -> None:
        raise ConnectionError("down")


class WarmingSpeaker(RecordingSpeaker):
    def __init__(self) -> None:
        super().__init__()
        self.rendered: list[str] = []

    async def warm(self, lines: list[str]) -> None:
        self.rendered.extend(lines)


async def test_warm_up_connects_renders_and_prefetches():
    looked.clear()
    pool, speaker = Pool(), WarmingSpeaker()
    warmup = Warmup(
        connections=[pool, Broken()],
        lines=["Hello, thanks for calling."],
        tools=[("account", {"caller": "ana"})],
    )
    acks = Acknowledgements(["Mm-hm."])
    answers = AnswerCache()
    answers.put("acme", "when do you open", "At nine.")
    agent, _ = agent_with(
        ScriptedModel([]), speaker, warmup=warmup, acks=acks, answers=answers, tenant="acme"
    )
    report = await agent.warm_up()
    assert pool.warmed == 1 and looked == ["ana"]
    assert set(speaker.rendered) == {"Hello, thanks for calling.", "Mm-hm.", "At nine."}
    assert acks.rendered == {"Mm-hm."}
    assert list(report.failed.values()) == ["ConnectionError"]
    await agent.executor.run("account", {"caller": "ana"}, agent.ctx)
    assert looked == ["ana"]  # the prefetch filled the cache


def test_prefetching_a_side_effect_is_refused():
    with pytest.raises(ToolUsageError):
        agent_with(ScriptedModel([]), warmup=Warmup(tools=[("charge", {"amount": 5})]))


async def test_a_turn_end_acknowledgement_plays_before_the_transcript_and_is_kept():
    async def slow(request):
        await asyncio.sleep(0.05)
        return "Here you go."

    acks = Acknowledgements(["Mm-hm."], fast_ms=20, expect_ms=500)
    agent, speaker = agent_with(CallableModel(slow), acks=acks)
    await agent.warm_up()
    agent.turn_ended()  # the caller stopped; the transcript is still on its way
    await asyncio.sleep(0)
    timing = TurnTiming(started=0.0)
    await agent.user_said("tell me", timing=timing)  # does not cut the acknowledgement
    await agent.wait()
    assert speaker.said == ["Mm-hm.", "Here you go."] and speaker.stops == 0
    assert timing.acked is not None


async def test_no_turn_end_acknowledgement_when_a_cached_answer_is_coming():
    answers = AnswerCache()
    answers.put("acme", "when do you open", "At nine.")
    acks = Acknowledgements(["Mm-hm."], fast_ms=0, expect_ms=500)
    agent, speaker = agent_with(ScriptedModel([]), acks=acks, answers=answers, tenant="acme")
    await agent.warm_up()
    agent.speculate("when do you open")
    agent.turn_ended()
    await agent.user_said("when do you open")
    await agent.wait()
    assert speaker.said == ["At nine."]


def test_the_example_runs(capsys):
    import runpy
    from pathlib import Path

    runpy.run_path(
        str(Path(__file__).parents[2] / "examples" / "voxcat_speed.py"), run_name="__main__"
    )
    out = capsys.readouterr().out
    assert "ran before the turn was final: ['weather Oslo']" in out
    assert "kept the early reply: True" in out
    assert "heard: ['Mm-hm.', 'Here you go.']" in out
    assert "cache hits: 1" in out
