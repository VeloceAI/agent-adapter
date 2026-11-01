"""Validated in-memory shape of a price table; prices are USD per 1M tokens."""

from __future__ import annotations

from datetime import date
from itertools import pairwise
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_adapter.core.types import ModelCaps

_STRICT = ConfigDict(frozen=True, extra="forbid")
_FLAT_FIELDS = ("input", "cached_input", "cache_write", "cache_write_by_ttl", "output")

Rate = Annotated[float, Field(ge=0)]  # USD per 1M tokens (or per unit for `units` entries)


class Tier(BaseModel):
    """Token prices that apply up to `max_prompt_tokens` prompt tokens (None: no limit)."""

    model_config = _STRICT

    max_prompt_tokens: int | None = Field(default=None, gt=0)
    input: Rate = Field(ge=0)
    cached_input: Rate | None = Field(default=None, ge=0)
    cache_write: Rate | None = Field(default=None, ge=0)
    cache_write_by_ttl: dict[str, Rate] = {}
    output: Rate = Field(ge=0)


class Schedule(BaseModel):
    """Token prices (flat or tiers), call and time prices, or `free`; plus cache storage."""

    model_config = _STRICT

    input: Rate | None = Field(default=None, ge=0)
    cached_input: Rate | None = Field(default=None, ge=0)
    cache_write: Rate | None = Field(default=None, ge=0)
    cache_write_by_ttl: dict[str, Rate] | None = None
    output: Rate | None = Field(default=None, ge=0)
    tiers: tuple[Tier, ...] | None = None
    cache_storage_per_hour: Rate | None = Field(default=None, ge=0)
    per_call: Rate | None = Field(default=None, ge=0)  # USD per call
    per_second: Rate | None = Field(default=None, ge=0)  # USD per second of call time
    free: bool = False  # costs nothing; never "no price"

    def has_flat(self) -> bool:
        """Whether any flat price field is set."""
        return any(getattr(self, name) is not None for name in _FLAT_FIELDS)

    def has_prices(self) -> bool:
        """Whether this schedule prices calls at all (a free one counts)."""
        return self.free or self._charges()

    def _charges(self) -> bool:
        timed = self.per_call is not None or self.per_second is not None
        return timed or self.tiers is not None or self.has_flat()

    def tier_list(self) -> tuple[Tier, ...]:
        """The tiers to choose from: the explicit tiers, one unlimited flat tier, or none."""
        if self.tiers is not None:
            return self.tiers
        if self.input is None or self.output is None:
            return ()
        return (
            Tier(
                input=self.input,
                cached_input=self.cached_input,
                cache_write=self.cache_write,
                cache_write_by_ttl=self.cache_write_by_ttl or {},
                output=self.output,
            ),
        )

    @model_validator(mode="after")
    def _check_prices(self) -> Schedule:
        if self.free and (self._charges() or self.cache_storage_per_hour is not None):
            raise ValueError("a free entry takes no prices")
        if self.tiers is not None and self.has_flat():
            raise ValueError("use either flat prices or tiers, not both")
        if self.has_flat() and (self.input is None or self.output is None):
            raise ValueError("flat prices need both input and output")
        if self.tiers is not None:
            _check_tiers(self.tiers)
        return self


def _check_tiers(tiers: tuple[Tier, ...]) -> None:
    if not tiers:
        raise ValueError("tiers need at least one tier")
    limits = [tier.max_prompt_tokens for tier in tiers]
    if limits[-1] is not None or None in limits[:-1]:
        raise ValueError("only the last tier may (and must) have max_prompt_tokens null")
    bounded = [limit for limit in limits[:-1] if limit is not None]
    if any(a >= b for a, b in pairwise(bounded)):
        raise ValueError("tiers must be in ascending max_prompt_tokens order")


class Period(Schedule):
    """A schedule in force from `from` to `until` (ISO dates, inclusive, UTC)."""

    start: date | None = Field(default=None, alias="from")
    until: date | None = None

    def covers(self, day: date) -> bool:
        """Whether `day` falls in this period."""
        return (self.start is None or self.start <= day) and (
            self.until is None or day <= self.until
        )

    @model_validator(mode="after")
    def _check_period(self) -> Period:
        if not self.has_prices():
            raise ValueError("each period needs prices or tiers")
        if self.start and self.until and self.until < self.start:
            raise ValueError(f"period until {self.until} is before from {self.start}")
        return self


class ModelEntry(Schedule):
    """One model row: provider, prices (flat, tiers or periods) and capability flags."""

    provider: str = Field(min_length=1)
    periods: tuple[Period, ...] | None = None
    caps: ModelCaps = ModelCaps()
    note: str = ""

    @field_validator("caps", mode="before")
    @classmethod
    def _known_caps(cls, value: Any) -> Any:
        if isinstance(value, dict):
            unknown = sorted(set(value) - set(ModelCaps.model_fields))
            if unknown:
                raise ValueError(f"unknown caps {unknown}")
        return value

    def caps_only(self) -> bool:
        """Whether the entry carries capability flags but no prices."""
        return self.periods is None and not self.has_prices()

    @model_validator(mode="after")
    def _check_periods(self) -> ModelEntry:
        if self.periods is None:
            return self
        if self.has_prices():
            raise ValueError("use either periods or top-level prices, not both")
        if not self.periods:
            raise ValueError("periods need at least one period")
        _check_overlap(self.periods)
        return self


def _check_overlap(periods: tuple[Period, ...]) -> None:
    lo, hi = date.min, date.max
    spans = sorted(((p.start or lo, p.until or hi) for p in periods), key=lambda s: s[0])
    for (_, end), (start, _) in pairwise(spans):
        if start <= end:
            raise ValueError(f"periods overlap around {start}")


class UnitEntry(BaseModel):
    """A non-token price: USD per billing unit, or `free`."""

    model_config = _STRICT

    provider: str = Field(min_length=1)
    per_unit: Rate | None = Field(default=None, ge=0)
    free: bool = False
    unit: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _check_rate(self) -> UnitEntry:
        if self.free and self.per_unit is not None:
            raise ValueError("a free unit entry takes no per_unit")
        if not self.free and self.per_unit is None:
            raise ValueError("a unit entry needs per_unit or free: true")
        return self


class Table(BaseModel):
    """A whole price table; `version` may be empty only in an override."""

    model_config = _STRICT

    version: str = ""
    currency: Literal["USD"] = "USD"
    unit: Literal["per_1M_tokens"] = "per_1M_tokens"
    note: str = ""
    aliases: dict[str, str] = {}
    models: dict[str, ModelEntry] = {}
    units: dict[str, UnitEntry] = {}

    @model_validator(mode="after")
    def _check_families(self) -> Table:
        # A family key matches any id after it, so it may only carry caps, never prices.
        for key, entry in self.models.items():
            if key.endswith("-") and not entry.caps_only():
                raise ValueError(f"family key {key!r} must be caps-only (no prices)")
        return self

    @model_validator(mode="after")
    def _check_patterns(self) -> Table:
        for kind, keys in (("model", self.models), ("unit", self.units)):
            for key in keys:
                if "*" in key[:-1]:
                    raise ValueError(f"{kind} key {key!r}: '*' may only end a pattern key")
        for alias, target in self.aliases.items():
            if "*" in alias or "*" in target:
                raise ValueError(f"alias {alias!r} -> {target!r}: aliases are exact, not a pattern")
        return self

    @model_validator(mode="after")
    def _check_units(self) -> Table:
        for item, entry in self.units.items():
            if item == "*":  # the catch-all: any provider
                continue
            if not item.startswith(f"{entry.provider}:") or item == f"{entry.provider}:":
                raise ValueError(f"unit key {item!r} must be 'provider:model' for {entry.provider}")
        return self


def check_aliases(table: Table) -> None:
    """Every alias must name a model in the table.

    Raises:
        ValueError: an alias points at a missing model.
    """
    for alias, target in table.aliases.items():
        if target not in table.models:
            raise ValueError(f"alias {alias!r} points at unknown model {target!r}")
