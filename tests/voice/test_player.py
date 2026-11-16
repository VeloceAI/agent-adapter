import asyncio

from agent_adapter.voice._speech import SpeechQueue
from agent_adapter.voice.media import CallableTTS
from agent_adapter.voice.player import Player
from agent_adapter.voice.playout import PlayoutBuffer
from agent_adapter.voice.testing import WordTTS, settle

MS = 32  # bytes per millisecond at 16 kHz


def player(ms_per_word: float = 100, rate: int = 16_000):
    tts = WordTTS(ms_per_word=ms_per_word, sample_rate=rate)
    return Player(tts, PlayoutBuffer(16_000)), tts


async def play(p: Player, ms: float) -> None:
    p.buffer.pull(int(ms * MS))
    await settle()


async def test_say_returns_when_the_line_has_played():
    p, _ = player()
    task = asyncio.create_task(p.say("two words"))
    await settle()
    assert p.speaking and not task.done()
    await play(p, 150)
    assert not task.done()
    await play(p, 60)
    assert task.done() and not p.speaking


async def test_prepare_synthesises_ahead_in_order():
    p, tts = player()
    p.prepare("one")
    p.prepare("two three")
    await settle()
    assert tts.spoken == ["one", "two three"]
    task = asyncio.create_task(p.say("one"))
    await settle()
    await play(p, 100)
    await task
    assert tts.spoken == ["one", "two three"]  # nothing synthesised twice


async def test_stop_records_the_words_heard_of_the_cut_line():
    p, _ = player()
    task = asyncio.create_task(p.say("one two three four"))
    await settle()
    await play(p, 210)  # just over half
    task.cancel()
    await p.stop()
    assert p.heard_of_cut == "one two"
    assert p.buffer.idle and not p.speaking


async def test_stop_cancels_prepared_lines():
    gate = asyncio.Event()

    async def slow(text: str):
        await gate.wait()
        return b"\x00\x01" * 10

    p = Player(CallableTTS(slow, sample_rate=16_000), PlayoutBuffer(16_000))
    p.prepare("later")
    await settle()
    await p.stop()
    assert p.heard_of_cut == "" and not p._ready


async def test_tts_audio_is_resampled_to_the_buffer_rate():
    p, _ = player(rate=24_000)
    task = asyncio.create_task(p.say("word"))
    await settle()
    await play(p, 99)
    assert not task.done()
    await play(p, 5)
    assert task.done()


async def test_a_failing_tts_ends_the_line_silently():
    def broken(text: str) -> bytes:
        raise RuntimeError("tts down")

    p = Player(CallableTTS(broken, sample_rate=16_000), PlayoutBuffer(16_000))
    await asyncio.wait_for(p.say("hello"), 1)


async def test_speech_queue_keeps_the_heard_part_of_a_cut_sentence():
    p, _ = player()
    speech = SpeechQueue(p)
    speech.feed("First one. Second sentence has five words. ")
    await settle()
    await play(p, 200)  # "First one." played
    await play(p, 300)  # 3 of 5 words
    await speech.stop()
    assert speech.take_played() == "First one. Second sentence has"


async def test_speech_queue_ignores_progress_lines_when_cut():
    p, _ = player()
    speech = SpeechQueue(p)
    speech.put("one moment please", record=False)
    await settle()
    await play(p, 200)
    await speech.stop()
    assert speech.take_played() == ""


async def test_a_repeated_short_line_comes_from_the_cache():
    p, tts = player()
    await p.warm(["One moment."])
    task = asyncio.create_task(p.say("One moment."))
    await settle()
    await play(p, 250)
    await task
    assert tts.spoken == ["One moment."]  # synthesised once


async def test_the_cache_keeps_only_the_last_lines_and_skips_long_ones():
    tts = WordTTS()
    p = Player(tts, PlayoutBuffer(16_000), cache_lines=2)
    await p.warm(["a", "b", "c", "x " * 100])
    assert list(p._cache) == ["b", "c"]
    off = Player(tts, PlayoutBuffer(16_000), cache_lines=0)
    await off.warm(["a"])
    assert not off._cache
