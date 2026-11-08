"""One superstep per call: the signal that ends a step, and a sink that keeps events whole."""

from __future__ import annotations

from agent_adapter.core.ports import EventSink
from agent_adapter.core.types import Event
from agent_adapter.tracing.events import EventType, emit


class SliceEnd(BaseException):
    """Unwinds the runner after a superstep's checkpoint is stored; carries the superstep count.

    Raised from `FlowRunner.on_superstep`, after the save, the event and the replay check.
    A BaseException so the runner's failure path (which marks the run failed) never sees it.
    """

    def __init__(self, steps: int) -> None:
        super().__init__(steps)
        self.steps = steps


class SliceSink:
    """Keeps one run's events whole across steps: no repeated `flow.started`, no empty recovery."""

    def __init__(self, inner: EventSink, *, first: bool) -> None:
        self._inner = inner
        self._first = first

    def emit(self, event: Event) -> None:
        if event.type == EventType.FLOW_STARTED and not self._first:
            return
        data = event.data
        if event.type == EventType.RUN_RECOVERED and not (
            data.get("replayed") or data.get("rerun")
        ):
            return  # a step that began on a clean checkpoint, not a crash
        emit(self._inner, event)

    @property
    def sinks(self) -> tuple[EventSink, ...]:
        """The sink behind this one, so audit checks, content and graphs reach it."""
        return (self._inner,)
