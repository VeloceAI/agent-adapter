"""Text views of case inputs and run outputs, shared by graders, reports and the optimiser."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from agent_adapter.evals.model import EvalCase


def clip(text: str, limit: int) -> str:
    """`text` cut to at most `limit` characters, with a marker saying how much was cut."""
    if len(text) <= limit:
        return text
    marker = f"\n[... {len(text) - limit + 31} chars clipped]"
    marker = f"\n[... {len(text) - (limit - len(marker))} chars clipped]"
    if limit <= len(marker):
        return text[:limit]
    return text[: limit - len(marker)] + marker


def as_text(value: Any) -> str:
    """A str as is, a pydantic model as JSON, anything else as sorted JSON."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def user_text(case: EvalCase) -> str:
    """The case input as the user turn: a str as is, anything else as JSON."""
    return as_text(case.input)


def output_text(run: object) -> str:
    """The run's output as text (`run.output`)."""
    return as_text(getattr(run, "output", ""))


def safe_text(text: str) -> str:
    """`text` that encodes as UTF-8: lone surrogates become `?`."""
    return text.encode("utf-8", "replace").decode("utf-8")


def normalised(text: str) -> str:
    """Lower-cased, whitespace collapsed, trailing `.,;:!?` removed."""
    return " ".join(text.casefold().split()).rstrip(".,;:!?").strip()
