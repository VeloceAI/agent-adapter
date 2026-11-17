# Voice calls and transports

Transports (sound card, WebSocket, phone networks, WebRTC), hosted and local speech, and call features such as recording, idle prompts and voicemail. Guide: [Voice call features](../how-to/voice-call-features.md).

## `agent_adapter.voice.observers`

`agent_adapter.voice.observers` re-exports, documented under the canonical path: [`CallObserver`](voice.md#agent_adapter.voice.CallObserver).

## `agent_adapter.voice.ending`

`agent_adapter.voice.ending` re-exports, documented under the canonical path: [`IdlePolicy`](voice.md#agent_adapter.voice.IdlePolicy), [`end_call_tool`](voice.md#agent_adapter.voice.end_call_tool).

## `agent_adapter.voice.recorder`

`agent_adapter.voice.recorder` re-exports, documented under the canonical path: [`CallRecorder`](voice.md#agent_adapter.voice.CallRecorder).

## `agent_adapter.voice.ambience`

`agent_adapter.voice.ambience` re-exports, documented under the canonical path: [`ThinkingSound`](voice.md#agent_adapter.voice.ThinkingSound).

## `agent_adapter.voice.voicemail`

::: agent_adapter.voice.voicemail
    options:
      members: [PHRASES, tone_of]

Also exported here, documented under the canonical path: [`VoicemailDetector`](voice.md#agent_adapter.voice.VoicemailDetector).

## `agent_adapter.voice.denoise`

`agent_adapter.voice.denoise` re-exports, documented under the canonical path: [`NoiseGate`](voice.md#agent_adapter.voice.NoiseGate).

## `agent_adapter.voice.local`

`agent_adapter.voice.local` re-exports, documented under the canonical path: [`LocalAudio`](voice.md#agent_adapter.voice.LocalAudio).

## `agent_adapter.voice.audio_socket`

::: agent_adapter.voice.audio_socket
    options:
      members: [BinarySocket, starlette_messages]

Also exported here, documented under the canonical path: [`AudioSocket`](voice.md#agent_adapter.voice.AudioSocket), [`binary_socket`](voice.md#agent_adapter.voice.binary_socket).

## `agent_adapter.voice.socket`

::: agent_adapter.voice.socket
    options:
      members: [TextSocket, VoiceSocket, text_socket]

## `agent_adapter.voice.twilio`

`agent_adapter.voice.twilio` re-exports, documented under the canonical path: [`TwilioStream`](voice.md#agent_adapter.voice.TwilioStream).

## `agent_adapter.voice.twilio_calls`

::: agent_adapter.voice.twilio_calls
    options:
      members: [dial_twiml, stream_twiml]

Also exported here, documented under the canonical path: [`TwilioCalls`](voice.md#agent_adapter.voice.TwilioCalls).

## `agent_adapter.voice.webrtc`

::: agent_adapter.voice.webrtc
    options:
      members: [OUT_RATE, WebRTCAudio, without_dtx]

## `agent_adapter.voice.webrtc_page`

::: agent_adapter.voice.webrtc_page
    options:
      members: [webrtc_page]

## `agent_adapter.voice.moonshine`

::: agent_adapter.voice.moonshine
    options:
      members: [MoonshineSTT, MoonshineTTS]

## `agent_adapter.voice.deepgram`

::: agent_adapter.voice.deepgram
    options:
      members: [Connect, DeepgramSTT, LiveSocket]

## `agent_adapter.voice.cloud`

::: agent_adapter.voice.cloud
    options:
      members: [CartesiaTTS, ElevenLabsTTS, OpenAISTT, OpenAITTS]
