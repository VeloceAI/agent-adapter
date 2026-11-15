---
title: "LiveKit voice agent tools that never book twice"
description: "Run your Python tools from LiveKit Agents with approvals, cost tracking and interruption-safe effects, or put a VoiceAgent on a LiveKit session."
---

# LiveKit

:material-flask-outline:{ .status-planned } Preview · raw-schema `function_tool` per spec ·
`pip install "agent-adapter[livekit]"` (`livekit-agents` 1.8 to below 2)

Contract tests run against the real `livekit-agents` 1.8.12 objects (`RawFunctionTool`,
`RunContext`, `SpeechHandle`); not yet run in a live room.

Two ways in:

- **Your tools in a LiveKit agent.** LiveKit's LLM decides; each tool call runs through
  the executor.
- **A `VoiceAgent` on a LiveKit session.** LiveKit hears and speaks; the
  [voice agent](../how-to/voice-agents.md) decides, with its cut rules and spoken approvals.

## Functions

| Function (`agent_adapter.integrations.livekit`) | Returns |
|---|---|
| `livekit_tool(spec, *, executor=None)` | one `RawFunctionTool` running `spec` through the executor |
| `livekit_tools(registry, *, executor=None)` | one per registry tool exported to `livekit` |
| `LiveKitSpeaker(session)` | a `Speaker` that says lines on an `AgentSession` |
| `attach_voice_agent(session, agent)` | sends the session's speech and final transcripts to `agent` |
| `INTEGRATION` | `get_integration("livekit")`; the call id comes from `RunContext.function_call` |

## Tools in a LiveKit agent

```python
from livekit.agents import Agent

from agent_adapter.integrations.livekit import livekit_tools

agent = Agent(instructions="You book museum tickets.", tools=livekit_tools(registry))
```

| When | What happens |
|---|---|
| A tool with `effects="additive"` or `"destructive"` starts | it calls `disallow_interruptions()`, so LiveKit keeps its result when the caller talks over the agent |
| The caller has already cut in | the tool is not started; the model gets a `ToolError` saying so |
| A `read_only` tool runs | LiveKit's own interruption rules apply |
| A tool calls `progress("One moment")` | `session.say` speaks it; it never enters the chat context |
| A tool error, denial, guardrail or run limit | the model gets a LiveKit `ToolError` with the kind and message |

## A VoiceAgent on a session

Start the `AgentSession` with STT, VAD and TTS but no `llm`; the `VoiceAgent` replies.

```python
from livekit.agents import AgentSession

from agent_adapter.integrations.livekit import LiveKitSpeaker, attach_voice_agent
from agent_adapter.voice import VoiceAgent

session = AgentSession(stt=stt, vad=vad, tts=tts)
agent = VoiceAgent(model, executor, LiveKitSpeaker(session))
attach_voice_agent(session, agent)
```

The speaker says each line with `allow_interruptions=False` and cuts with
`interrupt(force=True)`, so the voice agent alone decides when speech stops and the history
keeps only lines that played. Events reach the agent in order, without blocking LiveKit's
loop. Whether a session with no `llm` stays fully quiet at the end of a turn is not yet
checked on a live room.
