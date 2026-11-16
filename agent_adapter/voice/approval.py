"""Spoken approval: the agent asks the caller, and only that caller's next words decide."""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from agent_adapter.core.errors import ConfigError
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
MAX_SPOKEN_ARGS = 6
MAX_SPOKEN_VALUE = 48


def is_yes(text: str, words: frozenset[str] = YES_WORDS) -> bool:
    """True only when every word is a yes word: "yeah no", "yes but" and "" are all no."""
    tokens = _TOKEN.findall(text.lower().replace(chr(0x2019), "'"))  # curly apostrophe from STT
    return bool(tokens) and all(token in words for token in tokens)


Say = Callable[[str], Awaitable[None]]
Prompt = Callable[[ApprovalRequest], str]


def default_prompt(request: ApprovalRequest) -> str:
    """ "Shall I run refund with amount 9999? Say yes to confirm." (or from the hint)."""
    what = request.hint
    if not what:
        what = f"run {_spoken_name(request.tool)}"
        if request.arguments:
            what += f" with {describe_arguments(request.arguments)}"
    return f"Shall I {what}? Say yes to confirm."


def describe_arguments(arguments: Mapping[str, Any]) -> str:
    """Arguments in plain words ("amount 9999 and user bob"), long values cut short."""
    parts = [
        f"{_spoken_name(str(k))} {_spoken(v)}"
        for k, v in itertools.islice(arguments.items(), MAX_SPOKEN_ARGS)
    ]
    if len(arguments) > MAX_SPOKEN_ARGS:
        parts.append(f"{len(arguments) - MAX_SPOKEN_ARGS} more")
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def _spoken_name(name: str) -> str:
    return name.replace("_", " ")


def _spoken(value: Any, depth: int = 0) -> str:
    if isinstance(value, bool):
        text = "yes" if value else "no"
    elif value is None:
        text = "nothing"
    elif isinstance(value, Mapping | list | tuple):
        if depth >= 2:
            return "..."
        if isinstance(value, Mapping):
            first = itertools.islice(value.items(), MAX_SPOKEN_ARGS)
            parts = [f"{_spoken_name(str(k))} {_spoken(v, depth + 1)}" for k, v in first]
        else:
            parts = [_spoken(v, depth + 1) for v in value[:MAX_SPOKEN_ARGS]]
        text = ", ".join(parts) + (" ..." if len(value) > MAX_SPOKEN_ARGS else "")
    else:
        text = str(value)[: MAX_SPOKEN_VALUE * 4]
    text = " ".join(text.split())
    if len(text) <= MAX_SPOKEN_VALUE:
        return text
    return text[: MAX_SPOKEN_VALUE - 3].rstrip() + "..."


class _Line:
    """One call's side of the port: how to speak to it and its one open question."""

    def __init__(self, say: Say) -> None:
        self.say = say
        self.lock = asyncio.Lock()
        self.asking = False  # the prompt is still playing
        self.question: int | None = None  # set once the prompt has played
        self.answer: asyncio.Future[str] | None = None
        self.closed = False


class VoiceApproval:
    """Approval port for voice: asks the caller aloud, then waits for their own answer.

    Each call binds its `say` under its run id (a `VoiceAgent` or `RealtimeAgent` does
    this), so many calls can share one executor: a question goes only to the caller whose
    run asked, and only that caller's words answer it. A run with no bound call is denied.
    One question is open per call at a time, and an answer counts only when the caller
    began speaking after the prompt finished playing.

    Only text passed to `answer` (the caller's transcript) can approve; model text and tool
    output never reach it. Silence until `timeout_s`, or any answer that is not a clear yes,
    denies. `prompts` maps a tool name to its own prompt function.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 10.0,
        prompt: Prompt = default_prompt,
        yes_words: frozenset[str] = YES_WORDS,
        prompts: Mapping[str, Prompt] | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.prompt = prompt
        self.prompts = dict(prompts or {})
        self.yes_words = yes_words
        self._lines: dict[str | None, _Line] = {}
        self._ids = itertools.count(1)

    def bind(self, say: Say, *, run_id: str | None = None) -> None:
        """Speak this run's prompts with `say`. With no `run_id`, `say` serves every run that
        has no call of its own: only for a process that handles one call at a time."""
        line = self._lines.get(run_id)
        if line is not None and line.say != say:
            if run_id is not None:
                raise ConfigError(f"run {run_id!r} already has a voice call bound to it")
            self._close(line)
        if line is None or line.say != say:
            self._lines[run_id] = _Line(say)

    def unbind(self, run_id: str | None = None, say: Say | None = None) -> None:
        """Forget the run's call (when `say` is given, only if it is still the bound one);
        its open question is denied at once."""
        line = self._lines.get(run_id)
        if line is not None and (say is None or line.say == say):
            del self._lines[run_id]
            self._close(line)

    @property
    def waiting(self) -> bool:
        """True while any call has a question open."""
        return any(line.asking or line.answer is not None for line in self._lines.values())

    def waiting_for(self, run_id: str | None) -> bool:
        """True while the run's call has a question playing or waiting for its answer."""
        line = self._lines.get(run_id)
        return line is not None and (line.asking or line.answer is not None)

    def armed_question(self, run_id: str | None) -> int | None:
        """The id of the run's question once its prompt has played; None before that.

        Read it when the caller starts speaking and hand it to `answer` with their words:
        speech that began before the question was fully asked never answers it.
        """
        line = self._lines.get(run_id)
        return None if line is None else line.question

    def answer(self, text: str, *, run_id: str | None = None, question: int | None = None) -> bool:
        """Give the caller's final words to their open question; False when none is open.

        With `run_id`, the words count only if `question` is the id `armed_question` gave
        when the caller began them; others are taken but dropped. True means the words
        belonged to the open question, used or not.
        """
        line = self._lines.get(run_id)
        if line is None or not (line.asking or line.answer is not None):
            return False
        future = line.answer
        if future is None or future.done():
            return True
        if run_id is not None and (question is None or question != line.question):
            logger.info("voice approval: words that began before the question were dropped")
            return True
        future.set_result(text)
        return True

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        """Ask the run's caller aloud and decide from their answer; anything unclear denies."""
        line = self._lines.get(request.run_id) or self._lines.get(None)
        if line is None:
            return _deny("no voice call is bound to this run")
        async with line.lock:
            return await self._ask(line, request)

    async def _ask(self, line: _Line, request: ApprovalRequest) -> ApprovalDecision:
        if line.closed:
            return _deny("the call ended")
        future: asyncio.Future[str] | None = None
        line.asking = True
        try:
            await line.say(self.prompts.get(request.tool, self.prompt)(request))
            if line.closed:
                return _deny("the call ended")
            future = asyncio.get_running_loop().create_future()
            line.answer, line.question = future, next(self._ids)
            line.asking = False
            heard = await asyncio.wait_for(future, self.timeout_s)
        except TimeoutError:
            logger.info(
                "voice approval for tool %s timed out (call %s)", request.tool, request.call_id
            )
            return _deny("no answer")
        finally:
            line.asking = False
            if line.answer is future:
                line.answer, line.question = None, None
        if line.closed:
            return _deny("the call ended")
        if is_yes(heard, self.yes_words):
            return ApprovalDecision(approved=True, decided_by="voice", decided_at=time.time())
        return _deny("not a clear yes")

    @staticmethod
    def _close(line: _Line) -> None:
        line.closed = True
        if line.answer is not None and not line.answer.done():
            line.answer.set_result("")


def _deny(reason: str) -> ApprovalDecision:
    return ApprovalDecision(
        approved=False, reason=reason, decided_by="voice", decided_at=time.time()
    )
