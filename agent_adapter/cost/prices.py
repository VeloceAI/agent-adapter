"""The versioned, overridable price table: model resolution, token prices, unit prices, caps."""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.core.types import ModelCaps, Usage
from agent_adapter.cost._table import (
    ModelEntry,
    Schedule,
    Table,
    Tier,
    UnitEntry,
    check_aliases,
)

logger = logging.getLogger(__name__)

_PER = 1_000_000  # table prices are USD per 1M tokens
_STORAGE_SUFFIX = ":cache_storage"
# A prefix match counts only when what follows is a snapshot id, so `gpt-5.6-luna` is not
# priced as `gpt-5.6`, nor `claude-opus-4-5` as `claude-opus-4`. A caps-only key ending
# in "-" (e.g. "claude-") names a whole family instead.
_SNAPSHOT = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{3,}|latest)")
_WARNED: set[str] = set()
_WARN_LOCK = threading.Lock()
_NO_TOKENS = Tier(input=0.0, output=0.0)
# Match ranks on a tie of matched length: a priced row's snapshot beats a pattern, which
# beats a caps-only family.
_FAMILY, _PATTERN, _ROW = 0, 1, 2


def _warn_once(key: str, message: str, *args: Any) -> None:
    """Log `message` at WARNING the first time `key` is seen in this process."""
    with _WARN_LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    logger.warning(message, *args)


def _today() -> date:
    return datetime.now(UTC).date()


@dataclass(frozen=True)
class ResolvedPrice:
    """Token prices (USD per 1M tokens) for one model, tier and date."""

    model: str
    provider: str
    input: float
    cached_input: float
    cache_write: float
    cache_write_by_ttl: Mapping[str, float]
    output: float
    version: str
    per_call: float = 0.0
    per_second: float = 0.0
    free: bool = False


class Prices:
    """An immutable price table; build one with `default()`, `load()` or `merged()`."""

    __slots__ = ("_table",)

    def __init__(self, table: Table) -> None:
        self._table = table

    @property
    def version(self) -> str:
        """The table version recorded on priced ledger entries."""
        return self._table.version

    @classmethod
    def default(cls) -> Prices:
        """The packaged table (`cost/prices_default.json`), loaded once per process."""
        return _default_prices()

    @classmethod
    def empty(cls, version: str = "empty") -> Prices:
        """A table with no models, for pricing only your own with `merged()`.

        Raises:
            ConfigError: `version` is empty.
        """
        if not version:
            raise _invalid("empty", "version: a full table needs a non-empty version")
        return cls(Table(version=version))

    @classmethod
    def load(cls, source: Mapping[str, Any] | str | os.PathLike[str]) -> Prices:
        """A table from a mapping or a JSON file path.

        Raises:
            ConfigError: the file cannot be read, is not JSON, or the table is invalid.
        """
        data = _read(source)
        table = _parse(data, origin=_origin(source), aliases=True)
        if not table.version:
            raise _invalid(_origin(source), "version: a full table needs a non-empty version")
        return cls(table)

    def merged(self, overrides: Mapping[str, Any] | Prices) -> Prices:
        """A new table where each model, alias and unit in `overrides` replaces this one's.

        Raises:
            ConfigError: the overrides are invalid.
        """
        other = overrides._table if isinstance(overrides, Prices) else _parse(overrides)
        suffix = other.version or "custom"
        merged = Table.model_construct(
            version=f"{self.version}+{suffix}",
            currency=self._table.currency,
            unit=self._table.unit,
            note=self._table.note,
            aliases={**self._table.aliases, **other.aliases},
            models={**self._table.models, **other.models},
            units={**self._table.units, **other.units},
        )
        try:
            check_aliases(merged)
        except ValueError as exc:
            raise _invalid("overrides", str(exc)) from exc
        return Prices(merged)

    def resolve(self, model: str) -> str | None:
        """The table key for `model`: exact name, alias, then longest match; None if unknown.

        A row key matches when the rest is a snapshot suffix (`-2025-04-14`, `-20250514`,
        `-0613`, `-001`, `-latest`); a caps-only key ending in `-` matches its family; a
        key ending in `*` matches any id starting with what precedes it (`"*"`: any id).
        The longest matched prefix wins; on a tie, row, then pattern, then family. `caps()`
        reads this key; `price()` skips caps-only entries (`ResolvedPrice.model` is its key).
        """
        return self._resolve(model, priced=False)

    def _resolve_priced(self, model: str) -> str | None:
        """Like `resolve()`, but caps-only entries never match."""
        return self._resolve(model, priced=True)

    def _resolve(self, model: str, *, priced: bool) -> str | None:
        models = self._table.models

        def usable(key: str) -> bool:
            return not (priced and models[key].caps_only())

        if model in models and usable(model):
            return model
        alias = self._table.aliases.get(model)
        if alias is not None and usable(alias):
            return alias
        best: str | None = None
        best_rank = (-1, -1)
        for key, entry in models.items():
            if priced and entry.caps_only():
                continue  # a caps-only entry supplies caps but never blocks a price
            rank = _match_rank(key, entry, model)
            if rank is not None and rank > best_rank:
                best, best_rank = key, rank
        return best

    def price(
        self, model: str, *, at: date | None = None, prompt_tokens: int = 0
    ) -> ResolvedPrice | None:
        """Prices for `model` on `at` (default today, UTC) at this prompt size; None if unpriced."""
        key = self._resolve_priced(model)
        if key is None:
            self._warn_unpriced(model)
            return None
        entry = self._table.models[key]
        schedule = self._schedule(key, entry, at or _today())
        if schedule is None:
            return None
        tiers = schedule.tier_list()
        tier = _pick_tier(tiers, prompt_tokens) if tiers else _NO_TOKENS
        return ResolvedPrice(
            model=key,
            provider=entry.provider,
            input=tier.input,
            cached_input=tier.input if tier.cached_input is None else tier.cached_input,
            cache_write=tier.input if tier.cache_write is None else tier.cache_write,
            cache_write_by_ttl=MappingProxyType(dict(tier.cache_write_by_ttl)),
            output=tier.output,
            version=self.version,
            per_call=schedule.per_call or 0.0,
            per_second=schedule.per_second or 0.0,
            free=schedule.free,
        )

    def cost(
        self,
        model: str,
        usage: Usage,
        *,
        at: date | None = None,
        seconds: float = 0.0,
        attempts: int = 1,
    ) -> float | None:
        """USD for one call of `attempts` provider requests lasting `seconds`; None if unpriced.

        Raises:
            ConfigError: a per-second model got a negative or non-finite `seconds`, or a
                per-call model got `attempts` < 1.
        """
        price = self.price(model, at=at, prompt_tokens=usage.prompt_tokens)
        if price is None:
            return None
        if price.free:
            return 0.0
        if price.per_second and not (math.isfinite(seconds) and seconds >= 0):
            raise _invalid_call(model, f"seconds must be finite and >= 0, got {seconds!r}")
        if price.per_call and attempts < 1:
            raise _invalid_call(model, f"attempts must be >= 1, got {attempts!r}")
        labelled = sum(usage.cache_write_by_ttl.values())
        writes = sum(
            count * price.cache_write_by_ttl.get(ttl, price.cache_write)
            for ttl, count in usage.cache_write_by_ttl.items()
        )
        # Writes not split by TTL are charged at the default cache-write rate.
        writes += max(0, usage.cache_write_input_tokens - labelled) * price.cache_write
        tokens = (
            usage.input_tokens * price.input
            + usage.cached_input_tokens * price.cached_input
            + writes
            + usage.output_tokens * price.output
        )
        total = tokens / _PER
        if price.per_call:
            total += price.per_call * attempts  # each attempt is a billed request
        if price.per_second:
            total += price.per_second * seconds
        if usage.units:
            # A pattern key is not a model name: look the units up by the id called.
            name = model if price.model.endswith("*") else price.model
            total += self.unit_cost(f"{price.provider}:{name}", usage.units, at=at) or 0.0
        return total

    def unit_cost(self, item: str, units: float, *, at: date | None = None) -> float | None:
        """USD for `units` of `item` ("provider:model", or "provider:model:cache_storage").

        `item` is looked up exactly, then by the longest `*` pattern in `units`. A storage
        item uses its model's `cache_storage_per_hour` first (0 for a free model).
        """
        rate = None
        if item.endswith(_STORAGE_SUFFIX):
            rate = self._storage_rate(item.removesuffix(_STORAGE_SUFFIX), at or _today())
        if rate is None:
            entry = self._unit_entry(item)
            rate = None if entry is None else 0.0 if entry.free else entry.per_unit
        if rate is None:
            _warn_once(f"unit:{item}", "no unit price for %r; cost recorded as unknown", item)
            return None
        return units * rate

    def caps(self, model: str) -> ModelCaps:
        """Capability flags for `model`; defaults for an unknown model."""
        key = self.resolve(model)
        return ModelCaps() if key is None else self._table.models[key].caps

    def _warn_unpriced(self, model: str) -> None:
        caps_key = self.resolve(model)
        if caps_key is not None:
            _warn_once(f"caps:{caps_key}", "model %r has capabilities only, no price", caps_key)
            return
        _warn_once(f"model:{model}", "no price for model %r; cost recorded as unknown", model)

    def _schedule(self, key: str, entry: ModelEntry, day: date) -> Schedule | None:
        if entry.periods is None:
            return entry
        for period in entry.periods:
            if period.covers(day):
                return period
        _warn_once(f"period:{key}", "no price period for model %r on %s", key, day)
        return None

    def _unit_entry(self, item: str) -> UnitEntry | None:
        units = self._table.units
        if item in units:
            return units[item]
        matches = [key for key in units if key.endswith("*") and item.startswith(key[:-1])]
        return units[max(matches, key=len)] if matches else None

    def _storage_rate(self, provider_model: str, day: date) -> float | None:
        provider, _, model = provider_model.partition(":")
        key = self._resolve_priced(model) if model else None
        if key is None:
            return None
        entry = self._table.models[key]
        if entry.provider != provider:
            return None
        if entry.periods is None:
            return 0.0 if entry.free else entry.cache_storage_per_hour
        period = next((p for p in entry.periods if p.covers(day)), None)
        if period is not None and period.free:
            return 0.0
        if period is not None and period.cache_storage_per_hour is not None:
            return period.cache_storage_per_hour
        return entry.cache_storage_per_hour


def _match_rank(key: str, entry: ModelEntry, model: str) -> tuple[int, int] | None:
    """How well `key` matches `model` as (matched length, kind); None when it does not."""
    if key.endswith("*"):
        prefix = key[:-1]
        return (len(prefix), _PATTERN) if model.startswith(prefix) else None
    if not key or not model.startswith(key):
        return None
    if key.endswith("-") and entry.caps_only():
        return (len(key), _FAMILY)
    return (len(key), _ROW) if _SNAPSHOT.fullmatch(model[len(key) :]) is not None else None


def _pick_tier(tiers: tuple[Tier, ...], prompt_tokens: int) -> Tier:
    for tier in tiers:
        if tier.max_prompt_tokens is None or prompt_tokens <= tier.max_prompt_tokens:
            return tier
    return tiers[-1]  # unreachable for a validated table: the last tier is unlimited


def _origin(source: object) -> str:
    return "mapping" if isinstance(source, Mapping) else str(source)


def _read(source: Mapping[str, Any] | str | os.PathLike[str]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    if not isinstance(source, str | os.PathLike):
        raise _invalid(type(source).__name__, "source must be a mapping or a path")
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log_failure(logger, logging.ERROR, exc, "cannot read price table %s", path)
        raise ConfigError(f"cannot read price table {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _invalid(str(path), f"not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise _invalid(str(path), "the table must be a JSON object")
    return data


def _parse(data: Mapping[str, Any], *, origin: str = "overrides", aliases: bool = False) -> Table:
    try:
        table = Table.model_validate(dict(data))
        if aliases:
            check_aliases(table)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or '<table>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise _invalid(origin, problems) from exc
    except ValueError as exc:
        raise _invalid(origin, str(exc)) from exc
    return table


def _invalid_call(model: str, problem: str) -> ConfigError:
    logger.error("cannot price a call to %r: %s", model, problem)
    return ConfigError(f"cannot price a call to {model!r}: {problem}")


def _invalid(origin: str, problem: str) -> ConfigError:
    logger.error("invalid price table (%s): %s", origin, problem)
    return ConfigError(f"invalid price table ({origin}): {problem}")


@functools.cache
def _default_prices() -> Prices:
    return Prices.load(Path(__file__).with_name("prices_default.json"))
