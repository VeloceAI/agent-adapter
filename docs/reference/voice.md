# Voice agents

Turns, safe interruptions and spoken approval around the executor, our own audio pipeline and transports, plus offline fakes. Guide: [Build a voice agent](../how-to/voice-agents.md).

## `agent_adapter.voice`

::: agent_adapter.voice
    options:
      members: [AudioSocket, AudioTransport, CallObserver, CallRecorder, CallableSTT, CallableSpeaker, CallableTTS, EchoGuard, EventPump, IdlePolicy, LocalAudio, NoiseGate, Player, PlivoStream, STT, Speaker, TTS, TelnyxStream, ThinkingSound, TurnTiming, TwilioCalls, TwilioStream, VideoFeed, VoiceAgent, VoiceApproval, VoiceSession, VoicemailDetector, VonageStream, YES_WORDS, binary_socket, end_call_tool, is_yes, look_tool]

## `agent_adapter.voice.session`

::: agent_adapter.voice.session
    options:
      members: [BACKCHANNELS, RATE]

Also exported here, documented under the canonical path: [`TurnTiming`](#agent_adapter.voice.TurnTiming), [`VoiceSession`](#agent_adapter.voice.VoiceSession).

## `agent_adapter.voice.timing`

::: agent_adapter.voice.timing
    options:
      members: [STAGES, event_data]

Also exported here, documented under the canonical path: [`TurnTiming`](#agent_adapter.voice.TurnTiming).

## `agent_adapter.voice.speed`

::: agent_adapter.voice.speed
    options:
      members: [FastPath, Speed]

## `agent_adapter.voice.speculate`

::: agent_adapter.voice.speculate
    options:
      members: [ReplyOut, SPECULATIVE_STAGE, Speculation, SpeculationGate, SpeculativeTools, TOOL_ITEM, can_speculate, current, same_words]

## `agent_adapter.voice.ack`

::: agent_adapter.voice.ack
    options:
      members: [Acknowledgements]

## `agent_adapter.voice.answers`

::: agent_adapter.voice.answers
    options:
      members: [Answer, AnswerCache]

## `agent_adapter.voice.warm`

::: agent_adapter.voice.warm
    options:
      members: [WarmReport, Warmup, warm_up]

## `agent_adapter.voice.media`

::: agent_adapter.voice.media
    options:
      members: [STTStream, StreamingSTT]

Also exported here, documented under the canonical path: [`AudioTransport`](#agent_adapter.voice.AudioTransport), [`CallableSTT`](#agent_adapter.voice.CallableSTT), [`CallableTTS`](#agent_adapter.voice.CallableTTS), [`STT`](#agent_adapter.voice.STT), [`TTS`](#agent_adapter.voice.TTS).

## `agent_adapter.voice.player`

`agent_adapter.voice.player` re-exports, documented under the canonical path: [`Player`](#agent_adapter.voice.Player).

## `agent_adapter.voice.playout`

::: agent_adapter.voice.playout
    options:
      members: [Pacer, PlayoutBuffer, Segment, fire]

## `agent_adapter.voice.pcm`

::: agent_adapter.voice.pcm
    options:
      members: [Resampler, bytes_of, rms, samples_of, ulaw_decode, ulaw_encode]

## `agent_adapter.voice.turns`

::: agent_adapter.voice.turns
    options:
      members: [SilenceTurns, TurnDetector, TurnEvent, TurnTimer, pcm16]

Also exported here, documented under the canonical path: [`EchoGuard`](#agent_adapter.voice.EchoGuard).

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
      members: [CallerSaid, CallsCancelled, PromptDone, RealtimeAgent, RealtimeCodec, RealtimeEvent, SpeechStarted, ToolCalled]

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
      members: [MemoryTransport, RecordingSpeaker, ScriptedSTT, WordTTS, settle, silence, tone]
