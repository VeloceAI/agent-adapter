"""Voice agents: safe interruptions, spoken approval and one ledger, on any audio stack.

`VoiceSession` is our own pipeline (audio, turns, speech-to-text, text-to-speech); the
transports are `LocalAudio`, `AudioSocket`, `TwilioStream` and `WebRTCAudio`.
"""

from agent_adapter.voice.agent import VoiceAgent
from agent_adapter.voice.approval import YES_WORDS, VoiceApproval, is_yes
from agent_adapter.voice.audio_socket import AudioSocket, binary_socket
from agent_adapter.voice.local import LocalAudio
from agent_adapter.voice.media import STT, TTS, AudioTransport, CallableSTT, CallableTTS
from agent_adapter.voice.player import Player
from agent_adapter.voice.ports import CallableSpeaker, Speaker
from agent_adapter.voice.pump import EventPump
from agent_adapter.voice.session import TurnTiming, VoiceSession
from agent_adapter.voice.twilio import TwilioStream

__all__ = [
    "STT",
    "TTS",
    "YES_WORDS",
    "AudioSocket",
    "AudioTransport",
    "CallableSTT",
    "CallableSpeaker",
    "CallableTTS",
    "EventPump",
    "LocalAudio",
    "Player",
    "Speaker",
    "TurnTiming",
    "TwilioStream",
    "VoiceAgent",
    "VoiceApproval",
    "VoiceSession",
    "binary_socket",
    "is_yes",
]
