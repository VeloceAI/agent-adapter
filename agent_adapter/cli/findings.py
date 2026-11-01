"""One line of `agent-adapter doctor` output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Level = Literal["ok", "info", "warn", "error"]


@dataclass(frozen=True)
class Finding:
    """A check result: `level`, the area it is about, and what was found or what to do."""

    level: Level
    topic: str
    message: str

    def line(self) -> str:
        """`[warn ] topic: message`."""
        return f"[{self.level:<5}] {self.topic}: {self.message}"
