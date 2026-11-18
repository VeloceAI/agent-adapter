import asyncio
import time

import pytest

from agent_adapter import Registry
from agent_adapter.exec import Executor
from agent_adapter.voice import VoiceAgent
from agent_adapter.voice._speech import SpeechQueue
from agent_adapter.voice.endpoint import EagerTurns, looks_done
from agent_adapter.voice.session import VoiceSession
from agent_adapter.voice.speed import Speed
from agent_adapter.voice.testing import MemoryTransport, ScriptedSTT, WordTTS, settle, silence, tone
from agent_adapter.voice.turns import SilenceTurns
from tests.voice.test_session_speed import EchoModel, speak


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("Book a table for two.", True),
        ("What time is it?", True),
        ("yes", True),
        ("no thanks", True),
        ("I want to fly to Oslo and", False),
        ("Send it to the", False),
        ("my number is 98765 43210", False),
        ("Well, um", False),
        ("Let me think,", False),
        ("I want the blue one", None),
        ("", None),
    ],
)
def test_finished_and_trailing_words(text, verdict):
    assert looks_done(text) is verdict


def test_reading_the_words_costs_far_under_a_millisecond():
    texts = ["Book a table for two at seven tonight please.", "Send it to the"] * 500
    start = time.perf_counter()
    for text in texts:
        looks_done(text)
    assert (time.perf_counter() - start) / len(texts) < 0.0001


def feed(turns, pcm: bytes) -> list[tuple[int, str]]:
    events = []
    for i in range(0, len(pcm), 640):  # 20 ms frames
        events += [(i // 32, e) for e in turns.feed(pcm[i : i + 640])]
    return events


def test_finished_words_end_the_turn_sooner():
    turns = EagerTurns(SilenceTurns(start_ms=40, end_ms=800), done_ms=240)
    assert feed(turns, tone(300) + silence(100))[0][1] == "started"
    assert turns.hint("Book a table for two.") is True
    ended = feed(turns, silence(300))
    assert ended and ended[0][1] == "ended" and ended[0][0] <= 160  # 240 ms after speech


def test_trailing_words_hold_the_turn_open():
    turns = EagerTurns(SilenceTurns(start_ms=40, end_ms=800), hold_ms=1500)
    feed(turns, tone(300) + silence(100))
    turns.hint("I want to fly to Oslo and")
    assert feed(turns, silence(1000)) == []  # past the plain 800 ms
    assert feed(turns, silence(500))[0][1] == "ended"


def test_an_unsure_hint_keeps_the_detectors_own_wait():
    turns = EagerTurns(SilenceTurns(start_ms=40, end_ms=800))
    feed(turns, tone(300) + silence(100))
    turns.hint("I want the blue one")
    assert turns.timer.end_ms == 800


def test_speech_after_a_hint_resets_the_wait():
    turns = EagerTurns(SilenceTurns(start_ms=40, end_ms=800), done_ms=240)
    feed(turns, tone(300) + silence(100))
    turns.hint("Book a table.")
    feed(turns, tone(200))
    assert turns.timer.end_ms == 800 and turns.verdict is None


def test_a_caller_cut_off_too_soon_gets_a_longer_wait():
    turns = EagerTurns(SilenceTurns(start_ms=40, end_ms=800), done_ms=240)
    feed(turns, tone(300) + silence(100))
    turns.hint("Book a table.")
    feed(turns, silence(200) + tone(300))  # ended, then the caller went on at once
    assert turns.done_ms == 340


def test_long_mid_turn_pauses_raise_the_wait_for_that_caller():
    turns = EagerTurns(SilenceTurns(start_ms=40, end_ms=800), done_ms=240)
    feed(turns, tone(300) + (silence(400) + tone(300)) * 3 + silence(100))
    turns.hint("Book a table.")
    assert turns.timer.end_ms == 450  # the usual 400 ms pause, plus 50


def test_a_detector_without_a_timer_is_refused():
    class Bare:
        def feed(self, pcm):
            return []

    with pytest.raises(ValueError, match="TurnTimer"):
        EagerTurns(Bare())


async def test_the_session_hints_the_early_transcript_and_the_turn_ends_sooner():
    transport = MemoryTransport()
    turns = EagerTurns(SilenceTurns(start_ms=40, end_ms=800), done_ms=200)
    session = VoiceSession(
        transport,
        stt=ScriptedSTT(["Book a table."]),
        tts=WordTTS(ms_per_word=20),
        turns=turns,
        early_ms=60,
    )
    model = EchoModel()
    agent = VoiceAgent(model, Executor(Registry()), session.speaker, speed=Speed())
    run = asyncio.create_task(session.run(agent))
    await speak(transport, tone(300) + silence(300))
    assert turns.verdict is None and session.timings[0].ended is not None  # not 800 ms
    assert model.asked == ["Book a table."]
    transport.hang_up()
    await asyncio.wait_for(run, 2)


class Lines:
    def __init__(self) -> None:
        self.said: list[str] = []

    async def say(self, text: str) -> None:
        self.said.append(text)

    async def stop(self) -> None:
        return None


async def test_the_first_clause_is_spoken_before_the_sentence_ends():
    speaker = Lines()
    speech = SpeechQueue(speaker, first_words=4)  # type: ignore[arg-type]
    speech.feed("Well I can book that for you, ")
    await settle()
    assert speaker.said == ["Well I can book that for you,"]
    speech.feed("at seven tonight. Anything else? ")
    speech.flush()
    await speech.drain()
    assert speaker.said[1:] == ["at seven tonight.", "Anything else?"]


async def test_a_short_opener_goes_out_alone_and_the_next_line_is_eager_too():
    speaker = Lines()
    speech = SpeechQueue(speaker, first_words=4)  # type: ignore[arg-type]
    speech.feed("Sure, ")
    await settle()
    assert speaker.said == ["Sure,"]
    speech.feed("I can book that for you, ")
    await settle()
    assert speaker.said == ["Sure,", "I can book that for you,"]


async def test_a_long_first_clause_with_no_break_goes_out_after_twice_the_words():
    speaker = Lines()
    speech = SpeechQueue(speaker, first_words=4)  # type: ignore[arg-type]
    speech.feed("one two three four five six seven eight nine ")
    await settle()
    assert speaker.said == ["one two three four five six seven eight"]


async def test_each_reply_starts_fresh_and_without_first_words_it_waits_for_sentences():
    speaker = Lines()
    speech = SpeechQueue(speaker, first_words=4)  # type: ignore[arg-type]
    speech.feed("All booked for tonight.")
    speech.flush()
    speech.feed("Right then I will call them now, ")
    await settle()
    assert speaker.said == ["All booked for tonight.", "Right then I will call them now,"]
    plain = Lines()
    quiet = SpeechQueue(plain)  # type: ignore[arg-type]
    quiet.feed("Sure, I can book that for you, ")
    await settle()
    assert plain.said == []


class SlowStream:
    def __init__(self) -> None:
        self.partial = ""

    def feed(self, pcm: bytes) -> None:
        self.partial = "hello there"

    async def finish(self) -> str:
        await asyncio.sleep(0.3)
        return "hello there"

    async def cancel(self) -> None:
        return None


class SlowSTT:
    def open(self, sample_rate: int) -> SlowStream:
        return SlowStream()

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        return "hello there"


async def test_interim_words_start_the_reply_before_the_final_ones_arrive():
    transport = MemoryTransport()
    session = VoiceSession(
        transport,
        stt=SlowSTT(),
        tts=WordTTS(ms_per_word=20),
        turns=SilenceTurns(start_ms=40, end_ms=800),
        early_ms=60,
    )
    model = EchoModel()
    agent = VoiceAgent(model, Executor(Registry()), session.speaker, speed=Speed())
    run = asyncio.create_task(session.run(agent))
    await speak(transport, tone(300) + silence(100))
    assert model.asked == ["hello there"]  # the final words are still 0.3 s away
    transport.hang_up()
    await asyncio.wait_for(run, 2)
