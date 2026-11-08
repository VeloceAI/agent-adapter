"""The advisor's spend cap: hold each call's worst-case cost before it runs, settle after."""

from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from agent_adapter.core.errors import BudgetExceeded, ConfigError
from agent_adapter.core.messages import LLMRequest, LLMResponse, StreamDone, StreamEvent
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.types import Usage
from agent_adapter.cost.prices import Prices
from agent_adapter.evals._grading import Grader
from agent_adapter.evals.graders import all_of
from agent_adapter.evals.judge import LLMJudge
from agent_adapter.hooks.builtin.budgets.store import MemoryBudgetStore
from agent_adapter.runtime.ids import new_id

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = 4096  # each capped call's output limit when its request sets none
# Input a provider may add on its side (a tool-use system prompt, chat formatting).
INPUT_MARGIN_TOKENS = 2048
_KEY = "advisor"
_HOLD_S = 7 * 24 * 3600.0  # a hold never lapses while the advisor runs


class SpendCap:
    """A USD limit over every capped call: a worst-case hold first, the priced reply after.

    The worst case is every byte of the request as an input token (no tokenizer emits more
    tokens than UTF-8 bytes) plus `INPUT_MARGIN_TOKENS`, priced at the dearest input rate
    (plain or any cache write), plus `max_output_tokens` out; capped calls are sent with that
    output limit. A reply without usage keeps its whole hold. Retries inside a provider SDK
    are not seen: they run under one hold.
    """

    def __init__(
        self,
        max_usd: float | None,
        prices: Prices,
        *,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        self.max_usd = max_usd
        self.prices = prices
        self.max_output_tokens = max_output_tokens
        self.stopped = False
        self._store = MemoryBudgetStore()
        self._spent = 0.0

    @property
    def spent_usd(self) -> float:
        """What the settled calls cost."""
        return self._spent

    def check_priced(self, ports: Mapping[str, LLMPort]) -> None:
        """Fail closed: a capped run needs a token price for every model it may call.

        Raises:
            ConfigError: a model has no price, or is priced per second (no worst case).
        """
        if self.max_usd is None:
            return
        bad = sorted(name for name, port in ports.items() if not self._bounded(port.model))
        if bad:
            raise ConfigError(f"max_spend needs a token price for every model; none for {bad}")

    def limited(self, request: LLMRequest) -> LLMRequest:
        """`request` with its output limit at most `max_output_tokens` (unchanged uncapped)."""
        if self.max_usd is None:
            return request
        asked = request.max_output_tokens
        if asked is not None and asked <= self.max_output_tokens:
            return request
        return request.model_copy(update={"max_output_tokens": self.max_output_tokens})

    async def reserve(self, model: str, request: LLMRequest) -> tuple[str, float]:
        """Hold the call's worst-case cost: (hold id, amount); ("", 0) when uncapped.

        Raises:
            BudgetExceeded: the hold does not fit what is left; the cap is then stopped.
        """
        if self.max_usd is None:
            return "", 0.0
        worst = self.worst_usd(model, request)
        rid = new_id("adv_")
        held = await self._store.reserve(
            _KEY, rid, worst, limit_usd=self.max_usd, hold_s=_HOLD_S, expire_at=None
        )
        if self.stopped or not held.accepted:
            if held.accepted:
                await self._store.release(_KEY, rid)
            self.stopped = True
            logger.warning("advisor spend cap reached: a call's worst case would exceed it")
            raise BudgetExceeded(self.max_usd, held.usage.committed_usd, scope="advisor")
        return rid, worst

    async def settle(
        self, hold: tuple[str, float], response: LLMResponse | None, model: str
    ) -> None:
        """Replace the hold with the reply's real cost; without a reply the hold counts whole."""
        rid, worst = hold
        if response is None or _empty(response.usage):  # failed, cut short or unmetered
            cost = worst
        else:
            priced = self._cost(response.model, response.usage)
            cost = (priced if priced is not None else self._cost(model, response.usage)) or 0.0
            if rid and cost > worst:
                logger.warning("advisor: a reply cost more than its worst-case hold")
        self._spent += cost
        if rid:
            await self._store.settle(_KEY, rid, cost, expire_at=None)

    def worst_usd(self, model: str, request: LLMRequest) -> float:
        """The most `request` can cost on `model`, sent through `limited`."""
        size = len(request.model_dump_json().encode("utf-8")) + INPUT_MARGIN_TOKENS
        out = self.limited(request).max_output_tokens or self.max_output_tokens
        price = self.prices.price(model, prompt_tokens=size) if model else None
        ttls = tuple(price.cache_write_by_ttl) if price is not None else ()
        usages = [
            Usage(input_tokens=size, output_tokens=out),
            Usage(cache_write_input_tokens=size, output_tokens=out),
            *(
                Usage(
                    cache_write_input_tokens=size,
                    cache_write_by_ttl={ttl: size},
                    output_tokens=out,
                )
                for ttl in ttls
            ),
        ]
        costs = [c for c in (self._cost(model, u) for u in usages) if c is not None]
        return max(costs, default=0.0)

    def _bounded(self, model: str) -> bool:
        one = Usage(input_tokens=1, output_tokens=1)
        base = self._cost(model, one)
        return base is not None and self.prices.cost(model, one, seconds=60.0) == base

    def _cost(self, model: str, usage: Usage) -> float | None:
        return self.prices.cost(model, usage) if model else None


def _empty(usage: Usage) -> bool:
    return usage.prompt_tokens == 0 and usage.output_tokens == 0 and not usage.units


class CappedModel:
    """An `LLMPort` whose calls hold their worst-case cost in a `SpendCap` first."""

    def __init__(self, port: LLMPort, cap: SpendCap) -> None:
        self.port = port
        self.cap = cap
        self.provider = getattr(port, "provider", "")
        self.model = port.model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """The port's reply, once its worst case fits the cap.

        Raises:
            BudgetExceeded: the call's worst case would not fit.
            ModelError: as the port.
        """
        hold = await self.cap.reserve(self.model, request)
        response: LLMResponse | None = None
        try:
            response = await self.port.generate(self.cap.limited(request))
            return response
        finally:
            await self.cap.settle(hold, response, self.model)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """The port's stream, once its worst case fits the cap.

        Raises:
            BudgetExceeded: the call's worst case would not fit.
            ModelError: as the port.
        """
        hold = await self.cap.reserve(self.model, request)
        response: LLMResponse | None = None
        try:
            async for event in self.port.stream(self.cap.limited(request)):
                if isinstance(event, StreamDone):
                    response = event.response
                yield event
        finally:
            await self.cap.settle(hold, response, self.model)

    def cache_key_fields(self) -> dict[str, Any]:
        """The wrapped port's key material."""
        fields = getattr(self.port, "cache_key_fields", None)
        return dict(fields()) if callable(fields) else {}


def cap_graders(
    graders: Sequence[Grader] | Mapping[str, Grader], cap: SpendCap
) -> tuple[Sequence[Grader] | Mapping[str, Grader], dict[str, LLMPort]]:
    """The graders with every `LLMJudge` model (also inside `all_of`) behind `cap`.

    Returns the new graders and the judge models found, by a label, for `check_priced`.
    """
    judges: dict[str, LLMPort] = {}
    if isinstance(graders, Mapping):
        return {k: _capped(g, cap, judges, k) for k, g in graders.items()}, judges
    return [_capped(g, cap, judges, f"grader {i}") for i, g in enumerate(graders)], judges


def _capped(grader: Grader, cap: SpendCap, judges: dict[str, LLMPort], label: str) -> Grader:
    if isinstance(grader, LLMJudge):
        judges[f"judge of {label}"] = grader.model
        judge = copy.copy(grader)
        judge.model = CappedModel(grader.model, cap)
        judge._unguarded = judge.grade  # all_of calls this one
        return judge
    parts = getattr(grader, "parts", None)
    if getattr(grader, "__name__", "") == "all_of" and isinstance(parts, tuple):
        inner = [_capped(g, cap, judges, f"{label}[{i}]") for i, g in enumerate(parts)]
        if all(a is b for a, b in zip(inner, parts, strict=True)):
            return grader
        return all_of(*inner, weights=getattr(grader, "weights", None))
    return grader
