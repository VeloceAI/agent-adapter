"""Spoken approval: the agent asks, and only the caller's next words decide."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable

from agent_adapter.core.types import ApprovalDecision, ApprovalRequest

logger = logging.getLogger(__name__)

YES_WORDS = frozenset(
    {
        "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed", "correct",
        "right", "absolutely", "definitely", "please", "go", "ahead", "do", "it", "that's",
        "thats", "fine", "affirmative", "proceed",
    }
)  # fmt: skip
_TOKEN = re.compile(r"[a-z']+")


def is_yes(text: str, words: frozenset[str] = YES_WORDS) -> bool:
    """True only when every word is a yes word: "yeah no", "yes but" and "" are all no."""
    tokens = _TOKEN.findall(text.lower().replace(chr(0x2019), "'"))  # curly apostrophe from STT
    return bool(tokens) and all(token in words for token in tokens)


Say = Callable[[str], Awaitable[None]]


def default_prompt(request: ApprovalRequest) -> str:
    """ "Shall I book seat 4A? Say yes to confirm." from the tool's approval hint."""
    what = request.hint or f"run {request.tool.replace('_', ' ')}"
    return f"Shall I {what}? Say yes to confirm."


class VoiceApproval:
    """Approval port for voice: says a prompt, then waits for the caller's next final words.

    Only text passed to `answer` (the caller's transcript) can approve; model text and tool
    output never reach it. Silence until `timeout_s`, or any answer that is not a clear yes,
    denies. A `VoiceAgent` binds `say` and routes the caller's words here while it waits.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 10.0,
        prompt: Callable[[ApprovalRequest], str] = default_prompt,
        yes_words: frozenset[str] = YES_WORDS,
    ) -> None:
        self.timeout_s = timeout_s
        self.prompt = prompt
        self.yes_words = yes_words
        self._say: Say | None = None
        self._waiting: asyncio.Future[str] | None = None

    def bind(self, say: Say) -> None:
        """Set how the prompt is spoken (a `VoiceAgent` does this)."""
        self._say = say

    @property
    def waiting(self) -> bool:
        """True while a prompt waits for the caller's answer."""
        return self._waiting is not None and not self._waiting.done()

    def answer(self, text: str) -> bool:
        """Give the caller's final words to the waiting prompt; False when none waits."""
        if not self.waiting or self._waiting is None:
            return False
        self._waiting.set_result(text)
        return True

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        """Ask aloud and decide from the caller's answer; anything unclear denies."""
        if self._say is None:
            return _deny("voice approval is not bound to a voice agent")
        self._waiting = asyncio.get_running_loop().create_future()
        try:
            await self._say(self.prompt(request))
            heard = await asyncio.wait_for(asyncio.shield(self._waiting), self.timeout_s)
        except TimeoutError:
            logger.info(
                "voice approval for tool %s timed out (call %s)", request.tool, request.call_id
            )
            return _deny("no answer")
        finally:
            self._waiting = None
        if is_yes(heard, self.yes_words):
            return ApprovalDecision(approved=True, decided_by="voice", decided_at=time.time())
        return _deny("not a clear yes")


def _deny(reason: str) -> ApprovalDecision:
    return ApprovalDecision(
        approved=False, reason=reason, decided_by="voice", decided_at=time.time()
    )
