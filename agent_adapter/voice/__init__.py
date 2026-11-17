"""Voice agents: safe interruptions, spoken approval and one ledger, on any audio stack.

`VoiceSession` is our own pipeline (audio, turns, speech-to-text, text-to-speech); the
transports are `LocalAudio`, `AudioSocket`, `TwilioStream`, `TelnyxStream`, `PlivoStream`,
`VonageStream` and `WebRTCAudio`.
"""

from agent_adapter.voice.agent import VoiceAgent
from agent_adapter.voice.ambience import ThinkingSound
from agent_adapter.voice.approval import YES_WORDS, VoiceApproval, is_yes
from agent_adapter.voice.audio_socket import AudioSocket, binary_socket
from agent_adapter.voice.denoise import NoiseGate
from agent_adapter.voice.ending import IdlePolicy, end_call_tool
from agent_adapter.voice.local import LocalAudio
from agent_adapter.voice.media import STT, TTS, AudioTransport, CallableSTT, CallableTTS
from agent_adapter.voice.observers import CallObserver
from agent_adapter.voice.player import Player
from agent_adapter.voice.plivo import PlivoStream
from agent_adapter.voice.ports import CallableSpeaker, Speaker
from agent_adapter.voice.pump import EventPump
from agent_adapter.voice.recorder import CallRecorder
from agent_adapter.voice.session import TurnTiming, VoiceSession
from agent_adapter.voice.telnyx import TelnyxStream
from agent_adapter.voice.turns import EchoGuard
from agent_adapter.voice.twilio import TwilioStream
from agent_adapter.voice.twilio_calls import TwilioCalls
from agent_adapter.voice.video import VideoFeed, look_tool
from agent_adapter.voice.voicemail import VoicemailDetector
from agent_adapter.voice.vonage import VonageStream

__all__ = [
    "STT",
    "TTS",
    "YES_WORDS",
    "AudioSocket",
    "AudioTransport",
    "CallObserver",
    "CallRecorder",
    "CallableSTT",
    "CallableSpeaker",
    "CallableTTS",
    "EchoGuard",
    "EventPump",
    "IdlePolicy",
    "LocalAudio",
    "NoiseGate",
    "Player",
    "PlivoStream",
    "Speaker",
    "TelnyxStream",
    "ThinkingSound",
    "TurnTiming",
    "TwilioCalls",
    "TwilioStream",
    "VideoFeed",
    "VoiceAgent",
    "VoiceApproval",
    "VoiceSession",
    "VoicemailDetector",
    "VonageStream",
    "binary_socket",
    "end_call_tool",
    "is_yes",
    "look_tool",
]
