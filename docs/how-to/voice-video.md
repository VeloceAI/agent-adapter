---
title: "Let a voice agent see: camera, screen share and robot cameras"
description: "Give a VoiceSession agent the caller's camera or screen, a local camera or a robot camera: VideoFeed keeps the latest frame and a read-only look tool hands it to the model as an image."
---

# Let a voice agent see

The agent sees through a tool. `VideoFeed` keeps the latest frame from a camera, a screen
share or a robot. When the model calls `look`, it gets that frame as a JPEG image in the
tool result ([media in tool results](media-in-tool-results.md)), so it can answer "what is
on my screen?" or "what is in front of the robot?". Nothing is sent to the model unless it
looks.

```python
from agent_adapter.voice import VideoFeed, VoiceSession, look_tool
from agent_adapter.voice.webrtc import WebRTCAudio

feed = VideoFeed(fps=2)  # keeps one frame; JPEG only when the agent looks
registry.add(look_tool(feed, max_age=5))  # an older frame is an error, not "now"
browser = WebRTCAudio(video=feed)  # the caller's camera or screen share
session = VoiceSession(browser, stt=stt, tts=tts, turns=turns)
```

| Source | How | Needs |
|---|---|---|
| The caller's camera or screen, in a browser | `WebRTCAudio(video=feed)`; `webrtc_page()` has Camera and Share screen buttons | extra `webrtc` |
| A camera on this machine | `feed.camera()` | extra `webrtc` (PyAV) |
| This machine's screen | `feed.screen()` | extra `webrtc` (PyAV) |
| Any FFmpeg input: a file, a stream URL | `feed.capture(file, format=, options=)` | extra `webrtc` |
| A robot or simulator camera | `feed.poll(render)`: a function returning a frame | nothing extra |

A frame can be a PyAV frame, a numpy `uint8` array, a PIL image or encoded image bytes.
`look_tool` is read-only, so it is never asked about and is safe to repeat. A model that
cannot take images gets a note instead of the picture, so the call goes on.

When the browser turns its camera off, the server is not told, and the last frame stays.
Set `max_age=` so a stale frame raises a `ToolError` the model can act on.

## Measured, offline

Both checks use SmolVLM-256M-Instruct (Apache-2.0), a small vision model on the CPU, with
no account or key. It stands in for whatever vision model you use; the tool and the frames
are the same.

- `scripts/vision_check.py`: five drawn frames (coloured shapes and a word) go through
  `VideoFeed`, `look_tool` and the tool loop. It named 4 of 5. The miss was a green
  triangle it called just "a triangle". A tiny model answers one thing at a time: asking
  colour and shape together scored 1 of 5.
- `scripts/robot_vision.py`: a MuJoCo camera looks down at a red, green or blue block; the
  agent looks, the model names the colour, and the arm moves to that colour's bin through
  the [robot gate](robot-safety.md). It sorted 6 of 6, each colour twice.

The browser page was tried with fake camera and screen streams, not a real camera or a
screen picker; the two-peer test sends real video over aiortc in one process.

## What it does not do

- The agent has no face: avatar services are all paid, so there is no video out.
- It sees on request, not as a live stream: a model that watches video all the time
  (a realtime video model) needs that provider's own API.
