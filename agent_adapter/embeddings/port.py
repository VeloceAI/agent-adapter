"""The embedding port and the result every embedder returns."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, TypeAdapter, field_validator

from agent_adapter.core.context import ToolContext
from agent_adapter.core.types import Usage

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]  # int ok; str, bool not
Vector = Annotated[tuple[FiniteFloat, ...], Field(min_length=1)]
VECTOR: TypeAdapter[tuple[float, ...]] = TypeAdapter(Vector)


def plain(value: Any) -> Any:
    """`value.tolist()` for arrays and tensors (numpy, torch), else `value`."""
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


class EmbeddingResult(BaseModel):
    """Vectors in input order, the model that made them, its usage and the cache hits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vectors: tuple[Vector, ...]
    model: str = ""
    usage: Usage = Usage()
    cache_hits: NonNegativeInt = 0  # input texts served from the cache

    @field_validator("vectors", mode="before")
    @classmethod
    def _arrays_to_lists(cls, value: Any) -> Any:
        value = plain(value)
        if isinstance(value, list | tuple):
            return [plain(row) for row in value]
        return value


@runtime_checkable
class EmbeddingPort(Protocol):
    """Turns texts into vectors; raises `ModelError` subclasses."""

    provider: str
    model: str

    async def embed(
        self, texts: Sequence[str], *, ctx: ToolContext | None = None
    ) -> EmbeddingResult: ...
