"""A real WebRTC call in one process: a browser-like aiortc peer calls `WebRTCAudio`."""

import asyncio
import fractions

import pytest

aiortc = pytest.importorskip("aiortc")
av = pytest.importorskip("av")

from agent_adapter.voice.pcm import rms  # noqa: E402
from agent_adapter.voice.playout import PlayoutBuffer  # noqa: E402
from agent_adapter.voice.testing import tone  # noqa: E402
from agent_adapter.voice.webrtc import WebRTCAudio, without_dtx  # noqa: E402


class CallerTrack(aiortc.MediaStreamTrack):
    """The browser's microphone: a loud tone, in real time."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._pts = 0

    async def recv(self):
        await asyncio.sleep(0.02)
        frame = av.AudioFrame(format="s16", layout="mono", samples=960)
        frame.planes[0].update(tone(20, rate=48_000))
        frame.sample_rate, frame.pts = 48_000, self._pts
        frame.time_base = fractions.Fraction(1, 48_000)
        self._pts += 960
        return frame


def test_without_dtx():
    assert without_dtx("a=fmtp:111 minptime=10;usedtx=1") == "a=fmtp:111 minptime=10"


async def test_a_loopback_call_carries_audio_both_ways():
    rtc = WebRTCAudio()
    browser = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
    heard: list[bytes] = []

    async def listen(track) -> None:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)
        while True:
            try:
                received = await track.recv()
            except aiortc.mediastreams.MediaStreamError:
                return
            for frame in resampler.resample(received):
                heard.append(bytes(frame.planes[0])[: frame.samples * 2])

    browser.on("track", lambda track: asyncio.ensure_future(listen(track)))
    browser.addTrack(CallerTrack())
    await browser.setLocalDescription(await browser.createOffer())
    offer = {"sdp": browser.localDescription.sdp, "type": browser.localDescription.type}
    answer = await rtc.accept(offer)
    assert "usedtx=1" not in answer["sdp"]
    await browser.setRemoteDescription(aiortc.RTCSessionDescription(**answer))

    buffer = PlayoutBuffer(rtc.out_rate)
    await rtc.start(buffer)
    frames = rtc.frames()
    first = await asyncio.wait_for(anext(frames), 10)
    assert len(first) > 0
    caller = [first]
    for _ in range(20):
        caller.append(await asyncio.wait_for(anext(frames), 2))
    assert rms(b"".join(caller[-10:])) > 0.05  # the caller's tone came through Opus

    seg = buffer.open()
    buffer.write(seg, tone(600, rate=48_000))
    buffer.finish(seg)
    await asyncio.wait_for(seg.done, 3)
    await asyncio.sleep(0.3)
    assert max(rms(chunk) for chunk in heard) > 0.05  # the agent's voice reached the browser

    await browser.close()
    rest = [f async for f in frames]  # the call ends when the browser hangs up
    assert isinstance(rest, list)
    await rtc.close()
