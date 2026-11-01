"""Transcripts and metrics of one case run: tool-loop messages, flow events, hook activity."""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from agent_adapter.core.messages import LLMResponse, Message
from agent_adapter.core.types import Event, ToolCall
from agent_adapter.exec.tool_loop import ToolLoopResult
from agent_adapter.hooks.base import Hook, LLMInvocation, NodeInvocation, ToolInvocation, hook_name
from agent_adapter.hooks.builtin.logging import redact

logger = logging.getLogger(__name__)

# ToolContext.extras key of the per-run TranscriptRecorder (FlowHost sets it).
TRANSCRIPT_KEY = "aa.optimiser.transcript"


def render_call(call: ToolCall, *, withhold: Collection[str] = ()) -> str:
    """One `[ACTION]` line; secrets redacted, confirm-tool arguments withheld."""
    if call.name in withhold:
        return f"[ACTION] {call.name} (arguments withheld)"
    args = json.dumps(redact(call.arguments), ensure_ascii=False, sort_keys=True, default=str)
    return f"[ACTION] {call.name} {args}"


def render_messages(messages: Sequence[Message], *, withhold: Collection[str] = ()) -> str:
    """A tool-loop history in the transcript grammar; the system turn is left out."""
    lines: list[str] = []
    for message in messages:
        if message.role == "user":
            lines.append(f"[USER] {message.content}")
        elif message.role == "assistant":
            if message.content:
                lines.append(f"[ASSISTANT] {message.content}")
            lines += [render_call(call, withhold=withhold) for call in message.tool_calls]
        elif message.role == "tool":
            tag = "[OBSERVATION ERROR]" if _is_error(message.content) else "[OBSERVATION]"
            lines.append(f"{tag} {message.content}")
    return "\n".join(lines)


def _is_error(content: str) -> bool:
    if not content.startswith('{"error"'):
        return False
    try:
        value = json.loads(content)
    except ValueError:
        return False
    return isinstance(value, dict) and set(value) == {"error"}


def loop_metrics(result: ToolLoopResult) -> dict[str, float]:
    """Steps, tool use and token counts of one tool loop."""
    names = Counter(r.name for r in result.tool_results)
    metrics: dict[str, float] = {
        "steps": float(result.steps),
        "tool_calls": float(len(result.tool_results)),
        "tool_errors": float(sum(1 for r in result.tool_results if not r.ok)),
        "distinct_tools": float(len(names)),
        "tokens_in": float(result.usage.prompt_tokens),
        "tokens_out": float(result.usage.output_tokens),
    }
    metrics.update({f"tool_calls.{name}": float(n) for name, n in names.items()})
    return metrics


class TranscriptRecorder:
    """Collects one run's transcript lines from flow events and model replies (an EventSink)."""

    def __init__(self, *, withhold: Collection[str] = ()) -> None:
        self.lines: list[str] = []
        self.counts: Counter[str] = Counter()
        self._withhold = frozenset(withhold)
        self._lock = threading.Lock()

    def withhold_tools(self, names: Collection[str]) -> None:
        """Withhold the arguments of these tools from now on."""
        self._withhold = frozenset(names)

    def emit(self, event: Event) -> None:
        line = _event_line(event)
        with self._lock:
            self.counts[event.type] += 1
            if event.type == "tool.finished" or event.type == "tool.failed":
                self.counts[f"tool_calls.{event.data.get('tool', '')}"] += 1
            if line:
                self.lines.append(line)

    def add_reply(self, stage: str, prompt: str, response: LLMResponse) -> None:
        """Record one model call: the last user turn it saw and what it answered."""
        lines = [f"[INPUT] {stage}: {prompt}"] if prompt else []
        if response.text:
            lines.append(f"[ASSISTANT] {stage}: {response.text}")
        lines += [render_call(call, withhold=self._withhold) for call in response.tool_calls]
        with self._lock:
            self.counts["llm_calls"] += 1
            self.lines += lines

    def text(self) -> str:
        with self._lock:
            return "\n".join(self.lines)


def _event_line(event: Event) -> str:
    data: Mapping[str, Any] = event.data
    if event.type == "node.started":
        return f"[NODE] {data.get('node', '')} (visit {data.get('visit', '')})"
    if event.type == "node.failed":
        return f"[OBSERVATION ERROR] node {data.get('node', '')}: {data.get('error_kind', '')}"
    if event.type == "tool.finished":
        return f"[OBSERVATION] tool {data.get('tool', '')} finished"
    if event.type == "tool.failed":
        return f"[OBSERVATION ERROR] tool {data.get('tool', '')}: {data.get('error_kind', '')}"
    return ""


class TranscriptHook(Hook):
    """Records each model reply into the run's `TranscriptRecorder`, when the run has one."""

    name = "aa.transcript"

    async def after_llm(self, inv: LLMInvocation, response: LLMResponse) -> None:
        recorder = inv.ctx.extras.get(TRANSCRIPT_KEY)
        if isinstance(recorder, TranscriptRecorder):
            prompt = next(
                (m.content for m in reversed(inv.request.messages) if m.role == "user"), ""
            )
            recorder.add_reply(inv.request.stage, prompt, response)


class TrackedHook(Hook):
    """Delegates to `inner` and counts the calls where it acted (short-circuit, replace, raise)."""

    def __init__(self, inner: Hook) -> None:
        self.inner = inner
        self.acted = 0
        object.__setattr__(self, "name", hook_name(inner))

    async def _track(self, method: str, *args: Any) -> Any:
        try:
            value = await getattr(self.inner, method)(*args)
        except BaseException:
            self.acted += 1
            raise
        if value is not None:
            self.acted += 1
        return value

    async def before_tool(self, inv: ToolInvocation) -> Any:
        return await self._track("before_tool", inv)

    async def after_tool(self, inv: ToolInvocation, result: Any) -> Any:
        return await self._track("after_tool", inv, result)

    async def on_tool_error(self, inv: ToolInvocation, error: BaseException) -> Any:
        return await self._track("on_tool_error", inv, error)

    async def before_llm(self, inv: LLMInvocation) -> Any:
        return await self._track("before_llm", inv)

    async def after_llm(self, inv: LLMInvocation, response: LLMResponse) -> Any:
        return await self._track("after_llm", inv, response)

    async def on_llm_error(self, inv: LLMInvocation, error: BaseException) -> Any:
        return await self._track("on_llm_error", inv, error)

    async def before_node(self, inv: NodeInvocation) -> Any:
        return await self._track("before_node", inv)

    async def after_node(self, inv: NodeInvocation, update: Any) -> Any:
        return await self._track("after_node", inv, update)

    async def on_node_error(self, inv: NodeInvocation, error: BaseException) -> Any:
        return await self._track("on_node_error", inv, error)

    def config(self) -> dict[str, Any]:
        return self.inner.config()


def hook_metrics(hooks: Sequence[TrackedHook]) -> dict[str, float]:
    """`hook_fired.<name>`: how often each tunable hook acted during the run."""
    return {f"hook_fired.{hook_name(h)}": float(h.acted) for h in hooks}
