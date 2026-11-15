"""Voice agents: safe interruptions, spoken approval and one ledger, on any audio stack."""

from agent_adapter.voice.agent import VoiceAgent
from agent_adapter.voice.approval import YES_WORDS, VoiceApproval, is_yes
from agent_adapter.voice.ports import CallableSpeaker, Speaker
from agent_adapter.voice.pump import EventPump

__all__ = [
    "YES_WORDS",
    "CallableSpeaker",
    "EventPump",
    "Speaker",
    "VoiceAgent",
    "VoiceApproval",
    "is_yes",
]
