"""VoxCat quality checks: fast, model-free guards on a call's audio and words.

Each check is pure Python and costs well under a millisecond a frame or a sentence:
`InputCheck` (the caller's audio), `OutputCheck` (the agent's audio), `KnownNames` (fix
misheard names), `UnsureWords` (read back unsure numbers, dates and names before a
`confirm=True` tool) and `SpokenForm` (money, numbers and dates as a voice says them).
"""

from agent_adapter.voice.quality._input import InputCheck, TurnAudio
from agent_adapter.voice.quality._levels import FrameStats, measure
from agent_adapter.voice.quality._names import KnownNames, NameFix
from agent_adapter.voice.quality._numbers import number_words
from agent_adapter.voice.quality._output import Issue, OutputCheck
from agent_adapter.voice.quality._spoken import Currency, SpokenForm
from agent_adapter.voice.quality._stt import CheckedSTT, check_stt
from agent_adapter.voice.quality._unsure import Unsure, UnsureWords, Word

__all__ = [
    "CheckedSTT",
    "Currency",
    "FrameStats",
    "InputCheck",
    "Issue",
    "KnownNames",
    "NameFix",
    "OutputCheck",
    "SpokenForm",
    "TurnAudio",
    "Unsure",
    "UnsureWords",
    "Word",
    "check_stt",
    "measure",
    "number_words",
]
