"""The built-in chat agent: `run_tool_loop` over a model and a registry, text streamed."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence

from agent_adapter.core.messages import Message
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.exec.tool_loop import run_tool_loop
from agent_adapter.viewer.chat.contract import Attachment, ChatMessage, ChatSession

logger = logging.getLogger(__name__)

MAX_INLINE = 64 * 1024  # bytes of one text attachment put into the user message
_TEXT_TYPES = ("text/", "application/json")


class BuiltinAgent:
    """Answers with the model, running its tool calls through the session's executor."""

    def __init__(
        self,
        model: LLMPort,
        registry: Registry,
        *,
        system: str = "",
        max_steps: int = 8,
        stream: bool = True,
    ) -> None:
        self.model = model
        self.stream = stream
        self.registry = registry
        self.system = system
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return f"{self.model.provider}:{self.model.model}"

    def refusal(self, attachments: Sequence[Attachment]) -> str:
        """Why these attachments cannot be sent, or "" when they can."""
        for item in attachments:
            if not item.media_type.startswith(_TEXT_TYPES):
                return (
                    f"{item.name}: the built-in agent sends text only; "
                    "use --agent to handle images and PDF files"
                )
            if item.size > MAX_INLINE:
                return f"{item.name}: text attachments may hold {MAX_INLINE} bytes"
        return ""

    def messages(self, message: ChatMessage, session: ChatSession) -> list[Message]:
        """System prompt, earlier turns, then this message with its text attachments inlined.

        Raises:
            OSError, UnicodeDecodeError: an attachment cannot be read as UTF-8 text.
        """
        parts = [message.text]
        for item in message.attachments:
            parts.append(f"--- attachment: {item.name} ---\n{item.read_text()}")
        system = [Message(role="system", content=self.system)] if self.system else []
        user = Message(role="user", content="\n\n".join(part for part in parts if part))
        return [*system, *session.history, user]

    async def __call__(self, message: ChatMessage, session: ChatSession) -> AsyncIterator[str]:
        """Chunks as the model streams them; the whole answer at once when it cannot stream."""
        chunks: asyncio.Queue[str] = asyncio.Queue()
        streams = self.stream and callable(getattr(self.model, "stream", None))
        history = await asyncio.to_thread(self.messages, message, session)
        ctx = session.executor.context(run_id=session.run_id, state=session.state)
        loop = asyncio.ensure_future(
            run_tool_loop(
                self.model,
                self.registry,
                history,
                max_steps=self.max_steps,
                executor=session.executor,
                ctx=ctx,
                on_text=chunks.put_nowait if streams else None,
            )
        )
        getter: asyncio.Future[str] | None = None
        try:
            while not loop.done():
                getter = asyncio.ensure_future(chunks.get())
                done, _ = await asyncio.wait({getter, loop}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    yield getter.result()
            while not chunks.empty():
                yield chunks.get_nowait()
            result = loop.result()
        finally:
            for task in (getter, loop):
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        if not streams and result.final.text:
            yield result.final.text
        if result.finish != "stop":
            logger.info("chat: tool loop ended with %s (run %s)", result.finish, session.run_id)
            yield f"\n\n[the tool loop ended: {result.finish}]"
