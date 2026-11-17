"""`VideoFeed` and `look_tool`: let an agent see a browser camera, a screen or a robot camera.

The feed keeps only the latest frame, sampled at most `fps` times a second, and encodes it
to JPEG only when the agent looks. A frame is an `av.VideoFrame`, a uint8 RGB, RGBA or grey
array (numpy; a MuJoCo render is one), a PIL image, or encoded image bytes. Sources: a
WebRTC video track (`follow`, or `WebRTCAudio(video=feed)`), a local camera or the screen
(`camera`, `screen`; PyAV through aiortc), or any function that returns a frame (`poll`).
"""

from __future__ import annotations

import asyncio
import contextlib
import fractions
import inspect
import io
import logging
import math
import re
import sys
import time
from collections.abc import Callable
from typing import Any

from agent_adapter.core.errors import ConfigError, MissingDependency, ToolError
from agent_adapter.core.media import Media, sniff_mime
from agent_adapter.core.tool import ToolSpec, build_spec

logger = logging.getLogger(__name__)


def _av() -> Any:
    try:
        import av
    except ImportError as exc:
        raise MissingDependency(
            'video frames need PyAV: pip install "agent-adapter[webrtc]"'
        ) from exc
    return av


def to_jpeg(frame: Any, *, max_side: int = 1024, quality: int = 80) -> bytes:
    """`frame` as JPEG bytes, scaled down so its longest side is at most `max_side`.

    JPEG bytes are returned as they are; other encoded images are decoded and re-encoded.
    """
    encoded = bytes(frame) if isinstance(frame, bytes | bytearray | memoryview) else None
    if encoded is not None and sniff_mime(encoded) == "image/jpeg":
        return encoded
    av = _av()
    if encoded is not None:
        if not sniff_mime(encoded).startswith("image/"):
            raise ValueError("frame bytes must be an encoded image (JPEG, PNG, ...)")
        with av.open(io.BytesIO(encoded)) as container:
            video = next(container.decode(video=0))
    elif isinstance(frame, av.VideoFrame):
        video = frame
    elif type(frame).__module__.startswith("PIL."):
        rgb = frame if frame.mode in ("RGB", "RGBA") else frame.convert("RGB")
        video = av.VideoFrame.from_image(rgb)
    elif hasattr(frame, "__array_interface__"):
        video = _from_array(av, frame)
    else:
        raise TypeError(f"unsupported frame type {type(frame).__name__}")
    return _encode(av, video, max_side, quality)


def _from_array(av: Any, array: Any) -> Any:
    if str(array.dtype) != "uint8":
        raise ValueError(f"frame arrays must be uint8, not {array.dtype}")
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    formats = {2: "gray", 3: "rgb24", 4: "rgba"}
    channels = 2 if array.ndim == 2 else array.shape[2] if array.ndim == 3 else 0
    if channels not in formats:
        raise ValueError(f"frame array shape {array.shape} is not HxW, HxWx3 or HxWx4")
    if not array.flags["C_CONTIGUOUS"]:
        array = array.copy(order="C")
    return av.VideoFrame.from_ndarray(array, format=formats[channels])


def _encode(av: Any, frame: Any, max_side: int, quality: int) -> bytes:
    scale = min(1.0, max_side / max(frame.width, frame.height))
    width = max(2, round(frame.width * scale) // 2 * 2)
    height = max(2, round(frame.height * scale) // 2 * 2)
    codec = av.CodecContext.create("mjpeg", "w")
    codec.width, codec.height, codec.pix_fmt = width, height, "yuvj420p"
    codec.time_base = fractions.Fraction(1, 1)
    codec.qmin = codec.qmax = round(2 + (100 - quality) * 29 / 99)  # FFmpeg's 2 (best) .. 31
    resized = frame.reformat(width=width, height=height, format="yuvj420p")
    packets = [*codec.encode(resized), *codec.encode(None)]
    return b"".join(bytes(packet) for packet in packets)


class VideoFeed:
    """The latest frame of one video source, for `look_tool`. Start sources in async code."""

    def __init__(self, *, fps: float = 2.0, max_side: int = 1024, quality: int = 80) -> None:
        if fps <= 0 or max_side < 16 or not 1 <= quality <= 100:
            raise ConfigError("VideoFeed needs fps > 0, max_side >= 16 and quality 1..100")
        self.fps = fps
        self.max_side = max_side
        self.quality = quality
        self._frame: Any = None
        self._at: float | None = None
        self._seq = 0
        self._jpeg: tuple[int, bytes] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._players: list[Any] = []

    @property
    def latest(self) -> Any:
        """The latest kept frame, or None."""
        return self._frame

    def age(self) -> float | None:
        """Seconds since the latest frame was kept, or None with no frame."""
        return None if self._at is None else time.perf_counter() - self._at

    def push(self, frame: Any) -> bool:
        """Offer a frame; it is kept unless the last one was kept under 1/fps seconds ago."""
        now = time.perf_counter()
        if self._at is not None and now - self._at < 1 / self.fps:
            return False
        self._keep(frame, now)
        return True

    def clear(self) -> None:
        """Forget the latest frame (the camera was switched off)."""
        self._frame = self._at = self._jpeg = None

    def jpeg(self) -> bytes | None:
        """The latest frame as JPEG, encoded once per frame; None with no frame."""
        frame, seq, cached = self._frame, self._seq, self._jpeg
        if frame is None:
            return None
        if cached is not None and cached[0] == seq:
            return cached[1]
        data = to_jpeg(frame, max_side=self.max_side, quality=self.quality)
        self._jpeg = (seq, data)
        return data

    def follow(self, track: Any) -> asyncio.Task[None]:
        """Read a video track (aiortc `MediaStreamTrack`) until it ends."""
        return self._start(self._follow(track))

    def poll(self, source: Callable[[], Any]) -> asyncio.Task[None]:
        """Call `source` (sync or async) `fps` times a second; it returns a frame or None.

        A sync source runs on the event loop, so keep it quick (a MuJoCo render is); give a
        slow one as an async function, e.g. one that awaits `asyncio.to_thread`.
        """
        return self._start(self._poll(source))

    def capture(
        self, file: str, *, format: str | None = None, options: dict[str, str] | None = None
    ) -> asyncio.Task[None]:
        """Read any FFmpeg input (a device, a file, a stream URL) through aiortc's MediaPlayer."""
        try:
            from aiortc.contrib.media import MediaPlayer
        except ImportError as exc:
            raise MissingDependency(
                'VideoFeed.capture needs aiortc: pip install "agent-adapter[webrtc]"'
            ) from exc
        player = MediaPlayer(file, format=format, options=options)
        if player.video is None:
            raise ConfigError(f"{file!r} has no video stream")
        self._players.append(player)
        return self.follow(player.video)

    def camera(self, device: str | None = None) -> asyncio.Task[None]:
        """Read a local camera; `device` defaults to the first one this platform finds."""
        return self._device("camera", device)

    def screen(self, display: str | None = None) -> asyncio.Task[None]:
        """Read the screen (Windows, macOS, X11; not Wayland)."""
        return self._device("screen", display)

    def _device(self, kind: str, device: str | None) -> asyncio.Task[None]:
        file, format, options = device_input(kind, device, sys.platform, self.fps)
        return self.capture(file, format=format, options=options)

    async def close(self) -> None:
        """Stop every source this feed started."""
        players, self._players = self._players, []
        for player in players:
            player.video.stop()  # joins the player's thread before its loop can close
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _keep(self, frame: Any, now: float) -> None:
        self._frame, self._at = frame, now
        self._seq += 1

    def _start(self, work: Any) -> asyncio.Task[None]:
        task: asyncio.Task[None] = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _follow(self, track: Any) -> None:
        while True:
            try:
                frame = await track.recv()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # the track ended (MediaStreamError) or broke
                logger.debug("video: track ended (%s)", type(exc).__name__)
                return
            self.push(frame)

    async def _poll(self, source: Callable[[], Any]) -> None:
        failing = False
        while True:
            started = time.perf_counter()
            try:
                frame = source()
                if inspect.isawaitable(frame):
                    frame = await frame
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                level = logging.DEBUG if failing else logging.WARNING
                logger.log(level, "video: source failed (%s)", type(exc).__name__)
                failing = True
            else:
                failing = False
                if frame is not None:
                    self._keep(frame, time.perf_counter())
            await asyncio.sleep(max(0.0, 1 / self.fps - (time.perf_counter() - started)))


def device_input(
    kind: str, device: str | None, platform: str, fps: float
) -> tuple[str, str, dict[str, str]]:
    """The FFmpeg `(file, format, options)` for a camera or the screen on `platform`."""
    rate = {"framerate": str(max(1, math.ceil(fps)))}
    if kind == "screen":
        if platform == "win32":
            return device or "desktop", "gdigrab", rate
        if platform == "darwin":
            return device or "Capture screen 0", "avfoundation", rate
        return device or ":0.0", "x11grab", rate
    if platform == "win32":
        name = device or next(iter(_dshow_cameras()), None)
        if name is None:
            raise ConfigError("no camera found; pass camera(device=<DirectShow name>)")
        return f"video={name}", "dshow", {}
    if platform == "darwin":
        return device or "default", "avfoundation", {"framerate": "30"}
    return device or "/dev/video0", "v4l2", {}


def _dshow_cameras() -> list[str]:
    """Windows camera names, read from FFmpeg's device listing (it opens no camera)."""
    av = _av()
    previous = av.logging.get_level()
    av.logging.set_level(av.logging.INFO)
    try:
        with av.logging.Capture(True) as logs, contextlib.suppress(av.FFmpegError):
            av.open("dummy", format="dshow", options={"list_devices": "true"})
    finally:
        av.logging.set_level(previous)
    text = "".join(message for _, _, message in logs)
    return re.findall(r'"([^"]+)" \(video\)', text)


def look_tool(
    feed: VideoFeed,
    name: str = "look",
    *,
    description: str | None = None,
    max_age: float | None = None,
) -> ToolSpec:
    """A read-only tool that returns the feed's latest frame as a JPEG image.

    With `max_age`, a frame older than that many seconds counts as no frame: a video track
    that stopped sending (a camera switched off) stops being shown as the present.
    """

    def look() -> Media:
        age = feed.age()
        if age is not None and max_age is not None and age > max_age:
            raise ToolError(f"no new video frame for {age:.0f} s; the camera may be off", tool=name)
        data = feed.jpeg()
        if data is None:
            raise ToolError("no video frame yet; the camera or screen is not sending", tool=name)
        return Media(data=data, mime_type="image/jpeg", name=f"{name}.jpg")

    text = description or "See what the camera shows now: returns the latest video frame."
    return build_spec(look, name=name, description=text, effects="read_only", parallel_safe=True)


__all__ = ["VideoFeed", "device_input", "look_tool", "to_jpeg"]
