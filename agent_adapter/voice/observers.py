"""`CallObserver`: plug call features (recording, idle prompts, voicemail) into a `VoiceSession`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_adapter.voice.session import VoiceSession


class CallObserver:
    """Override what you need; every method is optional.

    `agent_audio` is called as the agent's audio plays, possibly on the sound card's
    thread: keep it quick. `caller_said` returning True keeps that text from the agent.
    """

    def started(self, session: VoiceSession) -> None:
        """The call started."""

    def caller_audio(self, pcm: bytes) -> None:
        """The caller's audio: 16 kHz PCM16 mono, after the session's filters."""

    def agent_audio(self, pcm: bytes) -> None:
        """The agent's audio as it plays, PCM16 mono at the transport's `out_rate`."""

    def caller_said(self, text: str) -> bool | None:
        """A final transcript of the caller; return True to keep it from the agent."""
        return None

    async def ended(self) -> None:
        """The call ended."""


__all__ = ["CallObserver"]
