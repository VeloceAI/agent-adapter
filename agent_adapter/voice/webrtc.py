"""`WebRTCAudio`: a browser call over WebRTC, peer to peer, with no media server (aiortc).

The browser sends an SDP offer to any HTTP route you own; `accept(offer)` returns the
answer. Audio is Opus at 48 kHz both ways. The browser's own echo cancellation keeps the
agent's voice out of the microphone, so speakers work, not only headphones.

With `video=VideoFeed(...)`, the browser's camera or screen share feeds it (see
`agent_adapter.voice.video`); without one, incoming video is declined.

Our answer never asks the browser for DTX: with DTX, the browser sends nothing in quiet,
and the first syllable after a pause (a "Wait") arrives clipped.
"""

from __future__ import annotations

import asyncio
import contextlib
import fractions
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import MissingDependency
from agent_adapter.voice.playout import PlayoutBuffer

if TYPE_CHECKING:
    from agent_adapter.voice.video import VideoFeed

logger = logging.getLogger(__name__)

OUT_RATE = 48_000
_FRAME_MS = 20
_ENDED = ("closed", "failed")


def _aiortc() -> tuple[Any, Any]:
    try:
        import aiortc
        import av
    except ImportError as exc:
        raise MissingDependency(
            'WebRTCAudio needs aiortc: pip install "agent-adapter[webrtc]"'
        ) from exc
    return aiortc, av


def without_dtx(sdp: str) -> str:
    """`sdp` with any Opus `usedtx=1` removed."""
    return re.sub(r";?\s*usedtx=1", "", sdp)


class WebRTCAudio:
    """An `AudioTransport` for one browser call; `accept` the browser's offer first."""

    out_rate = OUT_RATE

    def __init__(
        self,
        *,
        in_rate: int = 16_000,
        ice_servers: list[str] | None = None,
        video: VideoFeed | None = None,
    ) -> None:
        aiortc, av = _aiortc()
        self.in_rate = in_rate
        # aiortc falls back to a public STUN server; with no `ice_servers` we contact none
        # and offer host candidates only (a LAN, or a server with a public address).
        servers = [aiortc.RTCIceServer(urls=url) for url in ice_servers or ()]
        self.pc = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=servers))
        self._av = av
        self._out = _agent_track(aiortc, av)()
        self._frames: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._reader: asyncio.Task[None] | None = None
        self._video = video
        self._video_reader: asyncio.Task[None] | None = None
        self.pc.on("track", self._on_track)
        self.pc.on("connectionstatechange", self._on_state)

    async def accept(self, offer: dict[str, str]) -> dict[str, str]:
        """Answer the browser's `{"sdp", "type"}` offer; returns the answer to send back."""
        aiortc, _ = _aiortc()
        await self.pc.setRemoteDescription(
            aiortc.RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
        )
        if self._video is None:
            for transceiver in self.pc.getTransceivers():
                if transceiver.kind == "video":
                    transceiver.direction = "inactive"
        self.pc.addTrack(self._out)
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        local = self.pc.localDescription
        return {"sdp": without_dtx(local.sdp), "type": local.type}

    async def start(self, buffer: PlayoutBuffer) -> None:
        self._out.buffer = buffer

    async def frames(self) -> AsyncIterator[bytes]:
        while (frame := await self._frames.get()) is not None:
            yield frame

    async def close(self) -> None:
        for reader in (self._reader, self._video_reader):
            if reader is not None:
                reader.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reader
        self._reader = self._video_reader = None
        await self.pc.close()

    def _on_track(self, track: Any) -> None:
        if track.kind == "audio" and self._reader is None:
            self._reader = asyncio.ensure_future(self._read(track))
        elif track.kind == "video" and self._video is not None and self._video_reader is None:
            self._video_reader = self._video.follow(track)

    def _on_state(self) -> None:
        if self.pc.connectionState in _ENDED:
            self._frames.put_nowait(None)

    async def _read(self, track: Any) -> None:
        resampler = self._av.AudioResampler(format="s16", layout="mono", rate=self.in_rate)
        try:
            while True:
                frame = await track.recv()
                for mono in resampler.resample(frame):
                    self._frames.put_nowait(bytes(mono.planes[0])[: mono.samples * 2])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the browser's track ended (MediaStreamError) or broke
            logger.debug("webrtc: caller audio ended (%s)", type(exc).__name__)
        self._frames.put_nowait(None)


def _agent_track(aiortc: Any, av: Any) -> type:
    """The agent's outgoing track: pulls 20 ms from the playout buffer on each `recv`."""

    class AgentTrack(aiortc.MediaStreamTrack):  # type: ignore[misc]
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self.buffer: PlayoutBuffer | None = None
            self._samples = OUT_RATE * _FRAME_MS // 1000
            self._started: float | None = None
            self._pts = 0

        async def recv(self) -> Any:
            if self.readyState != "live":
                raise aiortc.mediastreams.MediaStreamError
            if self._started is None:
                self._started = time.monotonic()
            wait = self._started + self._pts / OUT_RATE - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            size = self._samples * 2
            pcm = self.buffer.pull(size) if self.buffer is not None else bytes(size)
            frame = av.AudioFrame(format="s16", layout="mono", samples=self._samples)
            frame.planes[0].update(pcm)
            frame.sample_rate = OUT_RATE
            frame.pts = self._pts
            frame.time_base = fractions.Fraction(1, OUT_RATE)
            self._pts += self._samples
            return frame

    return AgentTrack


__all__ = ["OUT_RATE", "WebRTCAudio", "without_dtx"]
