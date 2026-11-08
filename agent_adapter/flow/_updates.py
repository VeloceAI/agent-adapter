"""What one node visit produced: its updates, what hooks see, and whether the cache answered."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Updates:
    """A node visit's state updates (in order) and the view `after_node` hooks see."""

    items: list[Mapping[str, Any]] = field(default_factory=list)
    view: Mapping[str, Any] | None = None
    cached: bool = False  # the node cache answered; the action did not run
