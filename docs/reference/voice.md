# Voice agents

Turns, safe interruptions and spoken approval around the executor, on any audio stack, plus offline fakes. Guide: [Build a voice agent](../how-to/voice-agents.md).

## `agent_adapter.voice`

::: agent_adapter.voice
    options:
      members: [CallableSpeaker, EventPump, Speaker, VoiceAgent, VoiceApproval, YES_WORDS, is_yes]

## `agent_adapter.voice.socket`

::: agent_adapter.voice.socket
    options:
      members: [TextSocket, VoiceSocket, text_socket]

## `agent_adapter.voice.turns`

::: agent_adapter.voice.turns
    options:
      members: [SilenceTurns, TurnDetector, TurnEvent, TurnTimer, pcm16]

## `agent_adapter.voice.silero`

::: agent_adapter.voice.silero
    options:
      members: [SileroTurns]

## `agent_adapter.voice.smart_turn`

::: agent_adapter.voice.smart_turn
    options:
      members: [RATE, SmartTurn, log_mel]

## `agent_adapter.voice.realtime`

::: agent_adapter.voice.realtime
    options:
      members: [CallerSaid, CallsCancelled, RealtimeAgent, RealtimeCodec, RealtimeEvent, SpeechStarted, ToolCalled]

## `agent_adapter.voice.openai_realtime`

::: agent_adapter.voice.openai_realtime
    options:
      members: [OpenAIRealtime]

## `agent_adapter.voice.gemini_live`

::: agent_adapter.voice.gemini_live
    options:
      members: [GeminiLive]

## `agent_adapter.voice.testing`

::: agent_adapter.voice.testing
    options:
      members: [RecordingSpeaker, settle]
