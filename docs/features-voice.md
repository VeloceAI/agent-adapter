---
title: "Features: voice agents, video and robot safety"
description: "Voice calls with interruption-safe tools and spoken approvals, an agent that sees a camera or screen, and a safety gate for robots: what comes in the next release, with a guide for each."
---

# Features: voice, video and robots

These come in the next release; the rest is on [Features](features.md). How our own voice
stack compares with LiveKit Agents and Pipecat: [Voice stacks compared](how-to/voice-compared.md).

| Feature | Guide |
|---|---|
| Interruption-safe tools: a started payment or booking finishes once when the caller cuts in; a repeat window; spoken yes/no approvals bound to the call | [Build a voice agent](how-to/voice-agents.md) |
| `VoiceSession`: our own call pipeline with no media server; mic and speakers, WebSocket, browser WebRTC, Twilio, Telnyx, Plivo and Vonage; Silero VAD and Smart Turn; an echo guard | [A voice call with no framework](how-to/voice-pipeline.md), [Phone calls](how-to/voice-providers-and-phone.md), [Transports](how-to/voice-transports.md) |
| Speech: Moonshine offline, OpenAI, Deepgram, ElevenLabs, Cartesia, any provider via `CallableSTT` / `CallableTTS`; OpenAI Realtime and Gemini Live | [Hosted speech](how-to/voice-providers-and-phone.md#use-a-hosted-speech-provider), [Realtime models](how-to/voice-transports.md#openai-realtime-and-gemini-live) |
| Call features: idle prompt and hang-up, lines the caller cannot cut, mute while tools run, an end-call tool, typed input, recording, a thinking sound, voicemail detection, a noise filter | [Voice call features](how-to/voice-call-features.md) |
| LiveKit Agents and Pipecat connectors (preview): the same agent and safety rules inside their rooms and pipelines | [LiveKit](integrations/livekit.md), [Pipecat](integrations/pipecat.md), [Voice stacks compared](how-to/voice-compared.md) |
| Video: `VideoFeed` keeps the latest frame from a browser camera or screen share, a local camera, the screen or a robot camera; the read-only `look_tool` hands it to the model | [Let a voice agent see](how-to/voice-video.md) |
| Robot gate: workspace, speed and force limits checked before every move, a simulator shadow (MuJoCo), a latched emergency stop, an audit log; `SimRobot` and a ROS 2 port | [Gate a robot](how-to/robot-safety.md) |
