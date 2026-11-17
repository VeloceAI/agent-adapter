---
title: "Our voice stack vs LiveKit Agents vs Pipecat: features compared"
description: "Feature by feature: what our own voice pipeline does, what LiveKit Agents and Pipecat do, and where our LiveKit and Pipecat connectors are the answer."
---

# Voice stacks compared

We run voice calls three ways: our own [`VoiceSession`](voice-pipeline.md), or our
`VoiceAgent` inside LiveKit Agents or Pipecat through the connectors on
[Build a voice agent](voice-agents.md). The safety layer (finish-once tools, the repeat
window, spoken approvals, one ledger) is the same on all three. This page compares the rest.
Their columns come from their documentation, read on 2026-11-17.

✓ built in · ◐ partly, or through a paid service · ✗ not offered · → use our connector

| Feature | Ours | LiveKit Agents | Pipecat |
|---|---|---|---|
| **Safety** | | | |
| A started payment or booking finishes once when cut | ✓ | ◐ keeps the result | ✗ 84% left half-done in our test |
| Spoken yes/no approval bound to the call it was asked for | ✓ | ✗ | ✗ |
| Budgets, cost ledger, audit, one trace per call | ✓ | ◐ | ◐ |
| **Turns** | | | |
| Silero VAD, Smart Turn model | ✓ | ✓ own turn model | ✓ |
| Hush on speech, cut only on real speech | ✓ | ✓ | ✓ |
| "uh-huh" does not cut | ✓ | ✓ | ◐ |
| Lines the caller cannot cut; mute while tools run | ✓ | ✓ | ✓ |
| Echo guard for speakers with no echo cancelling | ✓ | ✗ | ✗ |
| History holds only the words the caller heard | ✓ | ✓ | ✗ |
| **Call features** | | | |
| Keypad (DTMF), transfer, outbound calls | ✓ Twilio | ✓ SIP | ✓ |
| Idle prompt and hang-up; end-call tool | ✓ | ✓ | ✓ |
| Voicemail detection | ✓ offline | ✓ | ◐ a model or paid classifier |
| IVR menu navigation | ✗ | ◐ | ✓ |
| Call recording | ✓ | ◐ egress service | ✓ |
| Thinking sound | ✓ | ✓ | ✓ |
| Noise filter | ◐ steady noise only | ◐ paid add-on | ◐ Krisp is paid |
| Typed input beside voice | ✓ | ✓ | ✓ |
| **Transports** | | | |
| Mic and speakers, WebSocket | ✓ | ◐ | ✓ |
| Browser WebRTC, one to one | ✓ | ✓ | ✓ |
| Rooms with many people | → LiveKit | ✓ | ◐ Daily, paid |
| Twilio | ✓ | ✓ | ✓ |
| Telnyx, Plivo, Vonage | ✓ format-tested | ✓ SIP | ✓ |
| SIP trunks | → LiveKit | ✓ | ◐ Daily |
| **Video** | | | |
| The agent sees the caller's camera or screen | ✓ `look` tool | ✓ | ✓ |
| A face (avatar) for the agent | ✗ | ◐ 16 paid vendors | ◐ paid vendors |
| **Around it** | | | |
| Ready-made speech providers | Moonshine, OpenAI, Deepgram, ElevenLabs, Cartesia; any via `CallableSTT`/`CallableTTS` | 70+ | 100+ |
| Realtime speech models | OpenAI Realtime, Gemini Live | 11 | ✓ |
| Mobile and web client SDKs | ✗ browser page only | ✓ 11 platforms | ✓ 6 platforms |
| Hosted cloud | ✗ | ✓ LiveKit Cloud | ✓ Pipecat Cloud |
| Runs with no account or key | ✓ | ✓ self-hosted | ✓ |

## Measured, not claimed

On one scripted call with the same models and detectors ([the race](voice-pipeline.md#the-same-call-on-three-stacks)):
our first reply came in 2.4 s against 4.3 s for both, we went quiet as fast as Pipecat
(54 ms) and faster than LiveKit (484 ms), and Pipecat used the least CPU. Charged once:
20 of 20 on every stack.

## When to use which

- **Ours**: one caller and one agent on a phone line, a browser or a laptop, where safety and
  first-reply speed matter most and you want no media server.
- **LiveKit through our connector**: rooms with many people, SIP trunks, mobile apps.
- **Pipecat through our connector**: a provider or transport we do not ship, IVR navigation,
  or a team that already runs Pipecat.
