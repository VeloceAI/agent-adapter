import asyncio
from types import SimpleNamespace

from agent_adapter import Registry
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.voice import VoiceAgent, VoiceApproval
from agent_adapter.voice.player import Player
from agent_adapter.voice.playout import PlayoutBuffer
from agent_adapter.voice.quality import (
    CheckedSTT,
    InputCheck,
    KnownNames,
    OutputCheck,
    SpokenForm,
    UnsureWords,
    Word,
    check_stt,
)
from agent_adapter.voice.session import VoiceSession
from agent_adapter.voice.testing import MemoryTransport, WordTTS, settle, silence, tone
from agent_adapter.voice.turns import SilenceTurns
from tests.fakes.models import ScriptedModel, calls, text

MS = 32


async def test_player_text_hooks_rewrite_what_is_synthesised():
    tts = WordTTS(ms_per_word=50)
    player = Player(tts, PlayoutBuffer(16_000), text_hooks=[SpokenForm()])
    player.prepare("Pay $5.")
    task = asyncio.create_task(player.say("Pay $5."))
    await settle()
    player.buffer.pull(200 * MS)
    await settle()
    await task
    assert tts.spoken == ["Pay five dollars."]  # prepared once, matched by its spoken form
    await player.warm(["At 9 am"])
    assert tts.spoken[-1] == "At nine A M"


async def test_a_broken_hook_leaves_the_line_and_the_cut_keeps_the_written_words():
    def broken(text: str) -> str:
        raise RuntimeError("boom")

    tts = WordTTS(ms_per_word=100)
    player = Player(tts, PlayoutBuffer(16_000), text_hooks=[broken, str.upper])
    task = asyncio.create_task(player.say("one two three four"))
    await settle()
    player.buffer.pull(210 * MS)
    await settle()
    task.cancel()
    await player.stop()
    assert tts.spoken == ["ONE TWO THREE FOUR"]
    assert player.heard_of_cut == "one two"


def request(arguments):
    return ApprovalRequest(
        approval_id="a", run_id="r1", call_id="c", tool="refund", arguments=arguments
    )


async def test_approval_puts_the_read_back_before_its_question():
    said: list[str] = []

    async def say(text: str) -> None:
        said.append(text)

    unsure = UnsureWords()
    approval = VoiceApproval(timeout_s=0.05, readback=unsure.readback)
    approval.bind(say, run_id="r1")
    unsure.hear([Word("4500", 0.4)], run_id="r1")
    await approval.request(request({"amount": 4500}))
    await approval.request(request({"amount": 4500}))
    assert said == [
        "Did you say 4500? Shall I run refund with amount 4500? Say yes to confirm.",
        "Shall I run refund with amount 4500? Say yes to confirm.",
    ]

    def broken(_: ApprovalRequest) -> str:
        raise RuntimeError("boom")

    approval.readback = broken
    decision = await approval.request(request({}))
    assert said[-1] == "Shall I run refund? Say yes to confirm." and not decision.approved


class WordsSTT:
    """Returns its transcripts with word confidences, as a hosted STT would."""

    def __init__(self, turns: list[list[Word]]) -> None:
        self.turns = turns
        self.last_words: list[Word] | None = None

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self.last_words = self.turns.pop(0) if self.turns else []
        return " ".join(w.text for w in self.last_words)


class Stream:
    def __init__(self) -> None:
        self.fed = 0
        self.cancelled = False
        self.last_words = [Word("Rahool", 0.99), Word("12", 0.3)]

    def feed(self, pcm: bytes) -> None:
        self.fed += len(pcm)

    async def finish(self) -> str:
        return "Rahool 12"

    async def cancel(self) -> None:
        self.cancelled = True


class StreamingWordsSTT(WordsSTT):
    def open(self, sample_rate: int) -> Stream:
        return Stream()


async def test_check_stt_fixes_names_and_notes_words_under_the_run():
    unsure = UnsureWords()
    checked = check_stt(
        WordsSTT([[Word("pay", 0.9), Word("Rahool", 0.99)]]),
        fixes=[KnownNames(["Rahul"])],
        unsure=unsure,
    )
    assert type(checked) is CheckedSTT
    checked.started(SimpleNamespace(agent=SimpleNamespace(ctx=SimpleNamespace(run_id="r1"))))
    assert await checked.transcribe(b"", 16_000) == "pay Rahul"
    assert [(i.text, i.kind) for i in unsure.unsure("r1")] == [("Rahul", "name")]  # a guess
    checked.started(SimpleNamespace(agent=None))
    assert checked.run_id is None

    streaming = check_stt(StreamingWordsSTT([]), fixes=[KnownNames(["Rahul"])], unsure=unsure)
    stream = streaming.open(16_000)  # type: ignore[attr-defined]
    stream.feed(b"\x00\x00")
    assert await stream.finish() == "Rahul 12"
    assert [i.text for i in unsure.unsure(None)] == ["12", "Rahul"]
    await stream.cancel()
    assert stream.stream.cancelled and stream.stream.fed == 2


async def test_a_call_with_every_check_reads_back_an_unsure_amount_in_spoken_form():
    registry = Registry()
    refunds: list[float] = []

    @registry.tool(confirm=True)
    async def refund(amount: float) -> str:
        """Refund an amount."""
        refunds.append(amount)
        return "refunded"

    unsure = UnsureWords()
    stt = check_stt(
        WordsSTT([[Word("refund", 0.99), Word("4500", 0.42)], [Word("yes", 0.99)]]), unsure=unsure
    )
    transport, tts = MemoryTransport(), WordTTS(ms_per_word=20)
    session = VoiceSession(
        transport,
        stt=stt,
        tts=tts,
        turns=SilenceTurns(start_ms=40, end_ms=200),
        filters=[gate := InputCheck(gate_db=-60)],
        observers=[stt, gate, SpokenForm(), OutputCheck()],
    )
    executor = Executor(registry, approval=VoiceApproval(timeout_s=5, readback=unsure.readback))
    model = ScriptedModel([calls(("refund", {"amount": 4500})), text("Refunded $4,500.")])
    run = asyncio.create_task(session.run(VoiceAgent(model, executor, session.speaker)))
    transport.speak(tone(400) + silence(260))
    await settle()
    for _ in range(100):
        transport.play(20)
        await settle(5)
    assert tts.spoken == [
        "Did you say four thousand five hundred? Shall I run refund with amount four thousand "
        "five hundred? Say yes to confirm."
    ]
    transport.speak(tone(150) + silence(260))
    await settle()
    for _ in range(100):
        transport.play(20)
        await settle(5)
    assert refunds == [4500] and tts.spoken[-1] == "Refunded four thousand five hundred dollars."
    transport.hang_up()
    await asyncio.wait_for(run, 2)
