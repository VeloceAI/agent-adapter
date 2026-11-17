"""`VideoFeed` and `look_tool` with fake frames: sampling, polling, encoding and the tool."""

import asyncio
import logging
import types

import pytest

import agent_adapter as aa
from agent_adapter.errors import ConfigError, ToolError
from agent_adapter.voice import video
from agent_adapter.voice.video import VideoFeed, device_input, look_tool, to_jpeg

JPEG = b"\xff\xd8\xff\xe0" + bytes(64)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(video, "time", types.SimpleNamespace(perf_counter=fake))
    return fake


def solid(rgb: tuple[int, int, int], width: int = 64, height: int = 48):
    av = pytest.importorskip("av")
    frame = av.VideoFrame(width, height, "rgb24")
    plane = frame.planes[0]
    row = bytes(rgb) * width
    plane.update((row + bytes(plane.line_size - len(row))) * height)
    return frame


def centre(jpeg: bytes) -> tuple[int, ...]:
    import io

    av = pytest.importorskip("av")
    with av.open(io.BytesIO(jpeg)) as container:
        frame = next(container.decode(video=0)).reformat(format="rgb24")
    plane = frame.planes[0]
    at = frame.height // 2 * plane.line_size + frame.width // 2 * 3
    return (frame.width, frame.height, *bytes(plane)[at : at + 3])


def look(feed: VideoFeed, **options) -> aa.Executor:
    registry = aa.Registry()
    registry.add(look_tool(feed, **options))
    return aa.Executor(registry)


def test_the_feed_keeps_only_the_latest_frame_at_most_fps_times_a_second(clock: Clock):
    feed = VideoFeed(fps=2)
    assert feed.latest is None and feed.age() is None
    assert feed.push("a")
    clock.now += 0.2
    assert not feed.push("b")  # under 1/fps after "a"
    assert feed.latest == "a"
    clock.now += 0.3
    assert feed.push("c")
    assert feed.latest == "c"
    clock.now += 1.5
    assert feed.age() == pytest.approx(1.5)
    feed.clear()
    assert feed.latest is None and feed.age() is None and feed.jpeg() is None


@pytest.mark.parametrize("options", [{"fps": 0}, {"max_side": 8}, {"quality": 101}])
def test_bad_settings_are_refused(options):
    with pytest.raises(ConfigError):
        VideoFeed(**options)


def test_jpeg_bytes_pass_through_without_pyav():
    feed = VideoFeed()
    feed.push(JPEG)
    assert feed.jpeg() == JPEG
    assert to_jpeg(bytearray(JPEG)) == JPEG


def test_a_frame_is_encoded_once_and_scaled_to_max_side(clock: Clock):
    feed = VideoFeed(max_side=32)
    feed.push(solid((255, 0, 0), 64, 48))
    first = feed.jpeg()
    assert first is not None and first[:3] == b"\xff\xd8\xff"
    assert feed.jpeg() is first  # cached until the next frame
    width, height, red, green, blue = centre(first)
    assert (width, height) == (32, 24)
    assert red > 200 and green < 60 and blue < 60
    clock.now += 1
    feed.push(solid((0, 0, 255)))
    assert centre(feed.jpeg())[4] > 200


def test_quality_trades_size_for_detail():
    av = pytest.importorskip("av")
    frame = av.VideoFrame(64, 64, "rgb24")
    frame.planes[0].update(bytes(range(256)) * (frame.planes[0].buffer_size // 256))
    assert len(to_jpeg(frame, quality=95)) > len(to_jpeg(frame, quality=10))


def test_arrays_pil_images_and_png_bytes_are_frames():
    np = pytest.importorskip("numpy")
    image = pytest.importorskip("PIL.Image")
    import io

    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[..., 1] = 255
    for frame in (rgb, rgb[::-1], rgb[..., :1], np.dstack([rgb, rgb[..., :1]])):
        assert to_jpeg(frame)[:3] == b"\xff\xd8\xff"
    png = io.BytesIO()
    image.fromarray(rgb).convert("P").save(png, "PNG")
    assert centre(to_jpeg(image.fromarray(rgb).convert("L")))[:2] == (60, 40)
    assert centre(to_jpeg(png.getvalue()))[3] > 200  # green survives the round trip
    with pytest.raises(ValueError, match="uint8"):
        to_jpeg(rgb.astype(np.float32))
    with pytest.raises(ValueError, match="shape"):
        to_jpeg(np.zeros((4, 4, 2), np.uint8))


def test_unknown_frames_are_refused():
    pytest.importorskip("av")
    with pytest.raises(ValueError, match="encoded image"):
        to_jpeg(b"not an image")
    with pytest.raises(TypeError, match="unsupported"):
        to_jpeg(object())


async def test_look_with_no_frame_yet_is_a_clear_tool_error():
    with pytest.raises(ToolError, match="no video frame yet"):
        await look(VideoFeed()).run("look", {})


async def test_look_returns_the_latest_frame_as_jpeg_media():
    feed = VideoFeed()
    spec = look_tool(feed, "glance", description="The wrist camera.")
    assert (spec.name, spec.effects) == ("glance", "read_only")
    assert spec.description == "The wrist camera."
    feed.push(JPEG)
    media = await look(feed).run("look", {})
    assert media == aa.Media(data=JPEG, mime_type="image/jpeg", name="look.jpg")


async def test_look_refuses_a_frame_older_than_max_age(clock: Clock):
    feed = VideoFeed()
    executor = look(feed, max_age=5)
    feed.push(JPEG)
    clock.now += 4
    assert (await executor.run("look", {})).data == JPEG
    clock.now += 2
    with pytest.raises(ToolError, match="camera may be off"):
        await executor.run("look", {})


async def test_the_tool_loop_shows_the_model_what_the_camera_sees():
    feed = VideoFeed()
    feed.push(JPEG)
    registry = aa.Registry()
    registry.add(look_tool(feed))
    seen: list[aa.Message] = []

    def model(request: aa.LLMRequest) -> dict:
        last = request.messages[-1]
        if last.role != "tool":
            return {"tool_calls": [{"name": "look", "arguments": {}}]}
        seen.append(last)
        return {"text": "a dark room"}

    ask = [aa.Message(role="user", content="What do you see?")]
    result = await aa.run_tool_loop(aa.CallableModel(model), registry, ask)
    assert result.final.text == "a dark room"
    assert seen[0].media == (aa.Media(data=JPEG, name="look.jpg"),)


async def test_poll_calls_sync_and_async_sources_and_survives_failures(caplog):
    feed = VideoFeed(fps=50)
    calls = 0

    def flaky():
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise RuntimeError("renderer not ready")
        return None if calls == 4 else f"frame {calls}"

    async def frames():
        return "async frame"

    with caplog.at_level(logging.DEBUG, logger="agent_adapter.voice.video"):
        feed.poll(flaky)
        for _ in range(200):
            if calls >= 6:
                break
            await asyncio.sleep(0.01)
    await feed.close()
    assert str(feed.latest).startswith("frame ")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "RuntimeError" in warnings[0].getMessage()

    other = VideoFeed(fps=50)
    task = other.poll(frames)
    for _ in range(100):
        if other.latest:
            break
        await asyncio.sleep(0.01)
    await other.close()
    assert other.latest == "async frame" and task.cancelled()


def test_device_inputs_per_platform(monkeypatch: pytest.MonkeyPatch):
    assert device_input("screen", None, "win32", 2) == ("desktop", "gdigrab", {"framerate": "2"})
    assert device_input("screen", None, "linux", 0.5)[:2] == (":0.0", "x11grab")
    assert device_input("screen", None, "darwin", 5)[1] == "avfoundation"
    assert device_input("camera", None, "linux", 2) == ("/dev/video0", "v4l2", {})
    assert device_input("camera", "Cam", "win32", 2) == ("video=Cam", "dshow", {})
    monkeypatch.setattr(video, "_dshow_cameras", lambda: ["USB Cam"])
    assert device_input("camera", None, "win32", 2)[0] == "video=USB Cam"
    monkeypatch.setattr(video, "_dshow_cameras", lambda: [])
    with pytest.raises(ConfigError, match="no camera"):
        device_input("camera", None, "win32", 2)
