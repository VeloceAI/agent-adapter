"""Ending calls: `IdlePolicy` checks on a silent caller; `end_call_tool` lets the agent hang up."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.voice.observers import CallObserver

if TYPE_CHECKING:
    from agent_adapter.voice.session import VoiceSession


class IdlePolicy(CallObserver):
    """When nobody has spoken for `after_s`, ask `prompt`; after `hang_up_after` unanswered
    prompts, say `goodbye` and hang up. Any caller words start the count again."""

    def __init__(
        self,
        *,
        after_s: float = 8.0,
        prompt: str = "Are you still there?",
        hang_up_after: int = 2,
        goodbye: str = "I'll hang up now. Goodbye.",
        check_s: float = 0.25,
    ) -> None:
        self.after_s = after_s
        self.prompt = prompt
        self.hang_up_after = hang_up_after
        self.goodbye = goodbye
        self.check_s = check_s
        self.prompts = 0
        self._task: asyncio.Task[None] | None = None

    def started(self, session: VoiceSession) -> None:
        self._task = asyncio.ensure_future(self._watch(session))

    def caller_said(self, text: str) -> None:
        self.prompts = 0

    async def ended(self) -> None:
        task, self._task = self._task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _watch(self, session: VoiceSession) -> None:
        quiet_since = time.monotonic()
        while True:
            await asyncio.sleep(self.check_s)
            agent, now = session.agent, time.monotonic()
            if agent is None or agent.busy or session.caller_talking or session.speaker.speaking:
                quiet_since = now
                continue
            if now - quiet_since < self.after_s:
                continue
            quiet_since = now
            if self.prompts >= self.hang_up_after:
                agent.say(self.goodbye, interruptible=False)
                await session.hang_up(after_speech=True)
                return
            self.prompts += 1
            agent.say(self.prompt, record=False)


def end_call_tool(
    session: VoiceSession,
    *,
    name: str = "end_call",
    description: str = "End the phone call once your reply has been spoken.",
) -> ToolSpec:
    """A tool for the agent to hang up after its goodbye: `registry.add(end_call_tool(session))`."""

    async def end_call() -> str:
        session.end_after_reply()
        return "The call ends after this reply."

    return build_spec(end_call, name=name, description=description, effects="additive")


__all__ = ["IdlePolicy", "end_call_tool"]
