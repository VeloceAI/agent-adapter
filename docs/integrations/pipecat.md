---
title: "Pipecat voice tools that never stop half-way"
description: "Register your Python tools on a Pipecat LLM service with approvals, cost tracking and interruption-safe effects, or put a VoiceAgent in a Pipecat pipeline."
---

# Pipecat

:material-flask-outline:{ .status-planned } Preview · `register_function` per spec ·
`pip install "agent-adapter[pipecat]"` (`pipecat-ai` 1.13 to below 2)

Contract tests run against the real `pipecat-ai` 1.16.0 classes (`LLMService`,
`FunctionCallParams`, `FrameProcessor`); not yet run in a live pipeline.

Two ways in:

- **Your tools on a Pipecat LLM service.** Pipecat's LLM decides; each call runs through
  the executor.
- **A `VoiceAgent` in the pipeline.** It takes the LLM's place:
  `stt -> processor -> tts`, with the [voice agent's](../how-to/voice-agents.md) cut rules
  and spoken approvals.

## Functions

| Function (`agent_adapter.integrations.pipecat`) | Returns |
|---|---|
| `pipecat_tools(llm, registry, *, executor=None)` | registers every tool exported to `pipecat` on `llm`; returns their `ToolsSchema` |
| `pipecat_tool(spec, *, executor=None)` | a `PipecatTool`: `schema`, `handler`, `cancel_on_interruption` and `register(llm)` |
| `voice_processor()` | a frame processor driving a `VoiceAgent` |
| `INTEGRATION` | `get_integration("pipecat")`; the call id is `FunctionCallParams.tool_call_id` |

## Tools on an LLM service

```python
from pipecat.processors.aggregators.llm_context import LLMContext

from agent_adapter.integrations.pipecat import pipecat_tools

tools = pipecat_tools(llm, registry)
context = LLMContext(messages, tools=tools)
```

| When | What happens |
|---|---|
| A tool with `effects="additive"` or `"destructive"` | registered with `cancel_on_interruption=False`: an interruption never stops it half-way |
| A `read_only` tool or one with no `effects` | cancelled on interruption, Pipecat's default |
| A tool calls `progress("One moment")` | pushed as a `TTSSpeakFrame` that stays out of the LLM context |
| A tool error, denial, guardrail or run limit | the result is `{"error": {"kind", "message"}}` |

## A VoiceAgent in the pipeline

```python
from pipecat.pipeline.pipeline import Pipeline

from agent_adapter.integrations.pipecat import voice_processor
from agent_adapter.voice import VoiceAgent

processor = voice_processor()
processor.use(VoiceAgent(model, executor, processor.speaker))
pipeline = Pipeline([transport.input(), stt, processor, tts, transport.output()])
```

`UserStartedSpeakingFrame` starts a cut and each `TranscriptionFrame` is a turn or an
approval answer; transcripts stop at the processor. Lines go out as `TTSSpeakFrame`s and a
cut broadcasts an interruption. Pipecat does not report playout per line, so after a cut
the history keeps the lines sent to TTS, which can be a little more than the caller heard.
