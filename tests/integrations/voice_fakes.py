"""Stand-in `livekit.agents` and `pipecat` modules: just the names the connectors touch."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_adapter.integrations import pipecat as pipecat_module


def _module(name: str, **names: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(names)
    module.__spec__ = None  # importable without a finder: already in sys.modules
    return module


# LiveKit


class LKToolError(Exception):
    pass


@dataclass
class RawTool:
    handler: Callable[..., Any]
    raw_schema: dict[str, Any]


def function_tool(handler: Callable[..., Any], *, raw_schema: dict[str, Any]) -> RawTool:
    return RawTool(handler, raw_schema)


@dataclass
class Handle:
    played: list[str]
    text: str

    async def wait_for_playout(self) -> None:
        self.played.append(self.text)


@dataclass
class Session:
    said: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    played: list[str] = field(default_factory=list)
    interrupts: list[bool] = field(default_factory=list)
    handlers: dict[str, Callable[[Any], None]] = field(default_factory=dict)

    def say(self, text: str, **options: Any) -> Handle:
        self.said.append((text, options))
        return Handle(self.played, text)

    async def interrupt(self, *, force: bool = False) -> None:
        self.interrupts.append(force)

    def on(self, event: str, callback: Callable[[Any], None]) -> None:
        self.handlers[event] = callback


@dataclass
class RunContext:
    session: Session = field(default_factory=Session)
    call_id: str = "lk_call_1"
    interrupted: bool = False
    locked: int = 0

    @property
    def function_call(self) -> Any:
        return types.SimpleNamespace(call_id=self.call_id)

    def disallow_interruptions(self) -> None:
        if self.interrupted:
            raise RuntimeError("speech already interrupted")
        self.locked += 1


def install_livekit(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _module("livekit.agents.llm", ToolError=LKToolError)
    agents = _module(
        "livekit.agents",
        function_tool=function_tool,
        llm=llm,
        RunContext=type("RunContext", (), {}),
    )
    root = _module("livekit", agents=agents)
    for module in (root, agents, llm):
        monkeypatch.setitem(sys.modules, module.__name__, module)


# Pipecat


@dataclass
class FunctionSchema:
    name: str
    description: str
    properties: dict[str, Any]
    required: list[str]


@dataclass
class ToolsSchema:
    standard_tools: list[Any]


@dataclass
class TTSSpeakFrame:
    text: str
    append_to_context: bool = True


@dataclass
class TranscriptionFrame:
    text: str


@dataclass
class InterimTranscriptionFrame:
    text: str


class UserStartedSpeakingFrame:
    pass


class FrameProcessor:
    def __init__(self, **kwargs: Any) -> None:
        self.pushed: list[tuple[Any, Any]] = []
        self.interruptions = 0

    async def process_frame(self, frame: Any, direction: Any) -> None:
        return None

    async def push_frame(self, frame: Any, direction: Any = "down") -> None:
        self.pushed.append((frame, direction))

    async def broadcast_interruption(self) -> None:
        self.interruptions += 1


@dataclass
class LLMService:
    registered: dict[str, tuple[Callable[..., Any], bool]] = field(default_factory=dict)
    pushed: list[Any] = field(default_factory=list)

    def register_function(
        self, name: str, handler: Callable[..., Any], *, cancel_on_interruption: bool
    ) -> None:
        self.registered[name] = (handler, cancel_on_interruption)

    async def push_frame(self, frame: Any) -> None:
        self.pushed.append(frame)


@dataclass
class Params:
    arguments: dict[str, Any]
    llm: LLMService
    tool_call_id: str = "pc_call_1"
    results: list[Any] = field(default_factory=list)

    async def result_callback(self, result: Any) -> None:
        self.results.append(result)


def install_pipecat(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _module(
        "pipecat.frames.frames",
        TTSSpeakFrame=TTSSpeakFrame,
        TranscriptionFrame=TranscriptionFrame,
        InterimTranscriptionFrame=InterimTranscriptionFrame,
        UserStartedSpeakingFrame=UserStartedSpeakingFrame,
    )
    modules = [
        _module("pipecat"),
        _module("pipecat.frames", frames=frames),
        frames,
        _module("pipecat.processors"),
        _module("pipecat.processors.frame_processor", FrameProcessor=FrameProcessor),
        _module("pipecat.adapters"),
        _module("pipecat.adapters.llm"),
        _module("pipecat.adapters.llm.schemas"),
        _module("pipecat.adapters.llm.schemas.function_schema", FunctionSchema=FunctionSchema),
        _module("pipecat.adapters.llm.schemas.tools_schema", ToolsSchema=ToolsSchema),
    ]
    for module in modules:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    pipecat_module._processor_class.cache_clear()
