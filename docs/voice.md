---
title: "Voice, video and robots: our own voice framework, LiveKit and Pipecat connectors, vision and robot safety"
description: "Agent Adapter's own voice framework (VoiceSession) for phone, browser and laptop calls, connectors for LiveKit Agents and Pipecat, an agent that sees a camera or screen, and a safety gate for robots."
---

# Voice, video and robots

Your agent can **talk** on a phone or browser call, **see** a camera or a shared screen, and
**move** a robot, with the same approvals, budgets, cost and traces as every other tool. All of
it comes in the next release; the rest of the SDK is on [Features](features.md).

<div class="grid cards" markdown>

-   :material-phone-in-talk:{ .lg .middle } **Our own voice framework**

    ---

    We built `VoiceSession`: phone, browser and laptop calls with no extra media server. It
    listens, knows when the caller has finished, answers, and stops when interrupted. A
    payment the caller talks over still happens only once, and approvals are a spoken "yes".

    [A voice call with no framework](how-to/voice-pipeline.md) · [Call features](how-to/voice-call-features.md)

-   :material-connection:{ .lg .middle } **Or inside LiveKit and Pipecat**

    ---

    Already use LiveKit Agents or Pipecat? The same agent and the same safety rules plug into
    them. Use them for rooms with many people, SIP trunks and mobile apps.

    [Build a voice agent](how-to/voice-agents.md) · [LiveKit](integrations/livekit.md) · [Pipecat](integrations/pipecat.md)

-   :material-video:{ .lg .middle } **Agents that can see**

    ---

    The caller turns on a camera or shares a screen, or a robot sends its camera. When the
    agent needs to, it takes one look and answers what it sees. Nothing is sent unless it looks.

    [Let a voice agent see](how-to/voice-video.md)

-   :material-robot-industrial:{ .lg .middle } **Robots that stay safe**

    ---

    Every move is checked against where the robot may go, how fast and how hard, before it
    runs. An emergency stop latches until a person resets it, and every action is audited.

    [Gate a robot](how-to/robot-safety.md)

</div>

## Start here

| I want to… | Read |
|---|---|
| answer phone calls (Twilio, Telnyx, Plivo, Vonage) | [Phone calls](how-to/voice-providers-and-phone.md) |
| talk to the agent in a browser or on my laptop | [A voice call with no framework](how-to/voice-pipeline.md), [Transports](how-to/voice-transports.md) |
| use hosted voices (OpenAI, Deepgram, ElevenLabs, Cartesia) or a realtime model | [Hosted speech](how-to/voice-providers-and-phone.md#use-a-hosted-speech-provider), [Realtime models](how-to/voice-transports.md#openai-realtime-and-gemini-live) |
| add idle prompts, recording, voicemail detection or a noise filter | [Voice call features](how-to/voice-call-features.md) |
| keep payments and bookings safe when the caller interrupts | [Build a voice agent](how-to/voice-agents.md) |
| choose between our stack, LiveKit and Pipecat | [Voice stacks compared](how-to/voice-compared.md) |
| let the agent see a camera or a screen | [Let a voice agent see](how-to/voice-video.md) |
| drive a robot or a simulator safely | [Gate a robot](how-to/robot-safety.md) |

## Measured, not claimed

All on one laptop, offline, with no account or key:

- **First reply:** 2.4 s on our stack against 4.3 s on LiveKit and Pipecat, on the same
  scripted call ([the race](how-to/voice-pipeline.md#the-same-call-on-three-stacks)).
- **Charged once:** 20 of 20 calls where the caller talks over a payment.
- **Voicemail:** 6 of 6 answering-machine greetings found, 0 of 8 people mistaken for one.
- **"Stop!" by voice:** a simulated arm halted in 470 ms (median of 5).
- **Seeing:** a small offline vision model sorted 6 of 6 coloured blocks with a simulated arm.
- **ROS 2:** a turtlesim robot moved, stopped and refused to move after the stop, 5 of 5.

## Everything in it

| Feature | Guide |
|---|---|
| Interruption-safe tools: a started payment or booking finishes once when the caller cuts in; a repeat window; spoken yes/no approvals bound to the call | [Build a voice agent](how-to/voice-agents.md) |
| `VoiceSession`: our own call pipeline with no media server; mic and speakers, WebSocket, browser WebRTC, Twilio, Telnyx, Plivo and Vonage; Silero VAD and Smart Turn; an echo guard | [A voice call with no framework](how-to/voice-pipeline.md), [Phone calls](how-to/voice-providers-and-phone.md), [Transports](how-to/voice-transports.md) |
| Speech: Moonshine offline, OpenAI, Deepgram, ElevenLabs, Cartesia, any provider via `CallableSTT` / `CallableTTS`; OpenAI Realtime and Gemini Live | [Hosted speech](how-to/voice-providers-and-phone.md#use-a-hosted-speech-provider), [Realtime models](how-to/voice-transports.md#openai-realtime-and-gemini-live) |
| Call features: idle prompt and hang-up, lines the caller cannot cut, mute while tools run, an end-call tool, typed input, recording, a thinking sound, voicemail detection, a noise filter | [Voice call features](how-to/voice-call-features.md) |
| LiveKit Agents and Pipecat connectors (preview): the same agent and safety rules inside their rooms and pipelines | [LiveKit](integrations/livekit.md), [Pipecat](integrations/pipecat.md), [Voice stacks compared](how-to/voice-compared.md) |
| Video: `VideoFeed` keeps the latest frame from a browser camera or screen share, a local camera, the screen or a robot camera; the read-only `look_tool` hands it to the model | [Let a voice agent see](how-to/voice-video.md) |
| Robot gate: workspace, speed and force limits checked before every move, a simulator shadow (MuJoCo), a latched emergency stop, an audit log; `SimRobot` and a ROS 2 port | [Gate a robot](how-to/robot-safety.md) |

Not built yet: phone menu (IVR) navigation is next on the [Roadmap](roadmap.md). Avatars,
multi-person rooms, SIP trunks and mobile apps come through the LiveKit and Pipecat connectors.
