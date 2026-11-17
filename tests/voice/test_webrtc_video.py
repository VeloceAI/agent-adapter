"""Real video in one process: a browser-like aiortc peer shows `WebRTCAudio` a camera."""

import asyncio
import fractions
import io

import pytest

aiortc = pytest.importorskip("aiortc")
av = pytest.importorskip("av")

import agent_adapter as aa  # noqa: E402
from agent_adapter.voice.video import VideoFeed, look_tool  # noqa: E402
from agent_adapter.voice.webrtc import WebRTCAudio  # noqa: E402


class CameraTrack(aiortc.MediaStreamTrack):
    """A browser camera that films a red wall, 15 frames a second."""

    kind = "video"

    def __init__(self) -> None:
        super().__init__()
        self._pts = 0

    async def recv(self):
        await asyncio.sleep(1 / 15)
        frame = av.VideoFrame(320, 240, "rgb24")
        plane = frame.planes[0]
        row = bytes((230, 20, 20)) * 320
        plane.update((row + bytes(plane.line_size - len(row))) * 240)
        frame.pts, frame.time_base = self._pts, fractions.Fraction(1, 90_000)
        self._pts += 6_000
        return frame


class SilentMic(aiortc.MediaStreamTrack):
    kind = "audio"

    async def recv(self):
        await asyncio.sleep(0.02)
        frame = av.AudioFrame(format="s16", layout="mono", samples=960)
        frame.planes[0].update(bytes(1920))
        frame.sample_rate, frame.time_base = 48_000, fractions.Fraction(1, 48_000)
        return frame


async def call(rtc: WebRTCAudio):
    browser = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
    browser.addTrack(SilentMic())
    browser.addTrack(CameraTrack())
    await browser.setLocalDescription(await browser.createOffer())
    offer = {"sdp": browser.localDescription.sdp, "type": browser.localDescription.type}
    answer = await rtc.accept(offer)
    await browser.setRemoteDescription(aiortc.RTCSessionDescription(**answer))
    return browser, answer["sdp"]


def video_section(sdp: str) -> str:
    return sdp[sdp.index("m=video") :]


def centre_rgb(jpeg: bytes) -> tuple[int, int, int]:
    with av.open(io.BytesIO(jpeg)) as container:
        frame = next(container.decode(video=0)).reformat(format="rgb24")
    plane = frame.planes[0]
    at = frame.height // 2 * plane.line_size + frame.width // 2 * 3
    red, green, blue = bytes(plane)[at : at + 3]
    return red, green, blue


async def test_look_returns_the_frame_the_browser_camera_sent():
    feed = VideoFeed(fps=5)
    rtc = WebRTCAudio(video=feed)
    registry = aa.Registry()
    registry.add(look_tool(feed))
    executor = aa.Executor(registry)
    browser, sdp = await call(rtc)
    assert "a=recvonly" in video_section(sdp)
    for _ in range(100):
        if feed.latest is not None:
            break
        await asyncio.sleep(0.1)
    media = await executor.run("look", {})
    assert media.mime_type == "image/jpeg"
    red, green, blue = centre_rgb(media.data)
    assert red > 180 and green < 80 and blue < 80  # the red wall, through VP8 and JPEG
    await browser.close()
    await rtc.close()


async def test_without_a_feed_the_browser_video_is_declined():
    rtc = WebRTCAudio()
    browser, sdp = await call(rtc)
    assert "a=inactive" in video_section(sdp)
    await browser.close()
    await rtc.close()


async def test_capture_reads_an_ffmpeg_input_through_media_player():
    feed = VideoFeed(fps=10)
    task = feed.capture("testsrc=size=160x120:rate=10", format="lavfi")
    for _ in range(100):
        if feed.latest is not None:
            break
        await asyncio.sleep(0.05)
    assert (feed.latest.width, feed.latest.height) == (160, 120)
    assert feed.jpeg()[:3] == b"\xff\xd8\xff"
    await feed.close()
    assert task.done()
