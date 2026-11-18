"""`OutputCheck`: clicks, clipping and dead air in the agent's own voice, as it plays."""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import log_failure
from agent_adapter.tracing.events import emit, make_event
from agent_adapter.voice.observers import CallObserver
from agent_adapter.voice.quality._levels import measure

if TYPE_CHECKING:
    from agent_adapter.voice.session import VoiceSession

logger = logging.getLogger(__name__)
_SEAM_S = 0.1


@dataclass(frozen=True)
class Issue:
    """One problem heard in the agent's audio."""

    kind: str  # "click", "clipping" or "silence"
    at: float  # time.monotonic()
    detail: float  # the jump, the clipped share, or the silent seconds


class OutputCheck(CallObserver):
    """Checks the agent's audio as it plays: a jump of `click_jump` (share of full scale)
    between two samples is a click, `clip_share` of a chunk at full scale is clipping, and
    `silence_s` of near-silent audio inside a line is dead air from a broken voice.

    It runs on the playout thread, so it only counts: `counts`, the last `keep` `issues`,
    and `on_issue(issue)` if given (keep that quick too). At the end of the call the counts go
    to the trace as a `custom.voice_quality` event. Pure Python, well under a millisecond a chunk.
    """

    def __init__(
        self,
        *,
        rate: int | None = None,
        click_jump: float = 0.6,
        clip_share: float = 0.01,
        silence_s: float = 1.5,
        silent_db: float = -60.0,
        keep: int = 50,
        on_issue: Callable[[Issue], None] | None = None,
    ) -> None:
        self.rate = rate
        self.click_jump = click_jump
        self.clip_share = clip_share
        self.silence_s = silence_s
        self.silent_db = silent_db
        self.on_issue = on_issue
        self.counts: Counter[str] = Counter()
        self.issues: deque[Issue] = deque(maxlen=keep)
        self._last: int | None = None
        self._last_at = 0.0
        self._silent_s = 0.0
        self._flagged = False
        self._session: VoiceSession | None = None

    def started(self, session: VoiceSession) -> None:
        self._session = session
        if self.rate is None:
            self.rate = session.transport.out_rate

    def agent_audio(self, pcm: bytes) -> None:
        self.feed(pcm)

    def feed(self, pcm: bytes) -> list[Issue]:
        """Check one chunk of the agent's audio; returns the issues it found."""
        now = time.monotonic()
        joined = now - self._last_at < _SEAM_S  # after a pause, a new line starts afresh
        stats = measure(pcm, previous=self._last if joined else None)
        if not stats.samples:
            return []
        end = stats.samples * 2
        self._last = int.from_bytes(pcm[end - 2 : end], "little", signed=True)
        self._last_at = now
        found: list[Issue] = []
        if stats.jump >= self.click_jump:
            found.append(Issue("click", now, stats.jump))
        if stats.clipped and stats.clipped >= stats.samples * self.clip_share:
            found.append(Issue("clipping", now, stats.clipped / stats.samples))
        if stats.level_db < self.silent_db:
            self._silent_s += stats.samples / (self.rate or 16_000)
            if self._silent_s >= self.silence_s and not self._flagged:
                self._flagged = True
                found.append(Issue("silence", now, self._silent_s))
        else:
            self._silent_s, self._flagged = 0.0, False
        for issue in found:
            self.counts[issue.kind] += 1
            self.issues.append(issue)
            if self.on_issue is not None:
                try:
                    self.on_issue(issue)
                except Exception as exc:
                    log_failure(logger, logging.WARNING, exc, "output check: on_issue failed")
        return found

    async def ended(self) -> None:
        agent = self._session.agent if self._session is not None else None
        if agent is not None and self.counts:
            data: dict[str, Any] = {"side": "agent", **self.counts}
            emit(agent.executor.sink, make_event("custom.voice_quality", agent.ctx, **data))
