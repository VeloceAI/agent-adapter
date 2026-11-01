"""Tool arguments as hooks, approvers and cache keys see them."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def dump_args(validated: BaseModel) -> dict[str, Any]:
    """Validated arguments with LLM-facing names, in JSON mode."""
    return validated.model_dump(mode="json", by_alias=True, fallback=str)
