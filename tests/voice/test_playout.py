import asyncio
import threading

from agent_adapter.voice.playout import Pacer, PlayoutBuffer
from agent_adapter.voice.testing import settle


async def test_a_line_is_done_once_its_last_byte_is_pulled():
    buffer = PlayoutBuffer(16_000)
    seg = buffer.open()
    buffer.write(seg, b"\x01" * 100)
    buffer.finish(seg)
    assert buffer.pull(60) == b"\x01" * 60
    await settle()
    assert not seg.done.done() and seg.heard == 0.6
    assert buffer.pull(60) == b"\x01" * 40 + bytes(20)
    await settle()
    assert seg.done.done() and buffer.idle


async def test_pause_gives_silence_and_resume_goes_on():
    buffer = PlayoutBuffer(16_000)
    seg = buffer.open()
    buffer.write(seg, b"\x02" * 40)
    buffer.pause()
    assert buffer.paused and buffer.pull(20) == bytes(20)
    assert buffer.pull(20, pad=False) == b""
    buffer.resume()
    assert buffer.pull(20) == b"\x02" * 20


async def test_clear_cuts_every_line_and_unpauses():
    buffer = PlayoutBuffer(16_000)
    cleared = []
    buffer.on_clear.append(lambda: cleared.append(True))
    seg = buffer.open()
    buffer.write(seg, b"\x03" * 40)
    buffer.pause()
    buffer.clear()
    await settle()
    assert seg.done.done() and buffer.idle and not buffer.paused and cleared
    buffer.write(seg, b"\x03")  # a cut line takes no more audio
    assert buffer.idle


async def test_an_empty_line_is_done_at_once():
    buffer = PlayoutBuffer(16_000)
    seg = buffer.open()
    buffer.finish(seg)
    await settle()
    assert seg.done.done()


async def test_confirmed_playout_waits_for_the_far_end():
    buffer = PlayoutBuffer(8_000, confirm_playout=True)
    drained, started = [], []
    buffer.on_drained.append(lambda s: drained.append(s.id))
    buffer.on_start.append(lambda s: started.append(s.id))
    seg = buffer.open()
    buffer.write(seg, b"\x04" * 10)
    buffer.finish(seg)
    buffer.pull(10)
    await settle()
    assert drained == [seg.id] and started == [seg.id] and not seg.done.done()
    buffer.confirm(seg.id)
    buffer.confirm(999)
    await settle()
    assert seg.done.done()


async def test_pull_from_another_thread_resolves_on_the_loop():
    buffer = PlayoutBuffer(16_000)
    seg = buffer.open()
    buffer.write(seg, b"\x05" * 64)
    buffer.finish(seg)
    thread = threading.Thread(target=lambda: buffer.pull(64))
    thread.start()
    thread.join()
    await asyncio.wait_for(seg.done, 1)


async def test_pacer_sends_in_real_time_and_resets_on_clear():
    buffer = PlayoutBuffer(8_000)
    sent: list[bytes] = []

    async def send(pcm: bytes) -> None:
        sent.append(pcm)

    pacer = Pacer(buffer, send, frame_ms=20, lead_ms=40)
    task = asyncio.create_task(pacer.run())
    seg = buffer.open()
    buffer.write(seg, b"\x06" * 8_000 * 2)  # 1 s of audio
    buffer.finish(seg)
    await asyncio.sleep(0.2)
    sent_ms = sum(len(p) for p in sent) / 16
    assert 150 <= sent_ms <= 320  # 200 ms played plus the lead, not the whole second
    buffer.clear()
    assert pacer._deadline == 0.0
    task.cancel()
