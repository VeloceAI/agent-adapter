"""One summariser call: the summary cache, the model request, its ledger entry and checks."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from agent_adapter.cache.codec import EnvelopeKind, decode, encode
from agent_adapter.cache.keys import KEY_VERSION, digest
from agent_adapter.context.entries import STAGE, SummaryEntry, next_entry_id, run_context
from agent_adapter.context.step import StepContext
from agent_adapter.context.summary_prompt import (
    PROMPT_VERSION,
    SUMMARY_SCHEMA,
    ContextSummary,
    data_message,
)
from agent_adapter.core.errors import ModelContentFiltered, ModelOutputInvalid, log_failure
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.ports import CachePort, LLMPort
from agent_adapter.core.types import ErrorInfo, Usage
from agent_adapter.runtime.current import use_context

logger = logging.getLogger(__name__)

NAMESPACE = "aa"
_KIND: EnvelopeKind = "llm"  # a summary is a model's output; its own key prefix keeps it apart


@dataclass(frozen=True)
class SummaryJob:
    """What to summarise, with which model and prompt, for which step settings."""

    step: str
    settings: Mapping[str, Any]  # the step's config(); part of the cache key
    model: LLMPort
    system_prompt: str
    covered: Sequence[Message]
    max_tokens: int
    cache: CachePort | None = None  # the step's own; else the policy's; never the run's
    ttl: float | None = None  # seconds a cached summary lives; None: no expiry


def summary_key(job: SummaryJob) -> str:
    """The summary's cache key over the covered messages and settings; "" if not serialisable."""
    provider, model = job.model.provider, job.model.model
    try:
        fields = job.model.cache_key_fields()
        material = [
            PROMPT_VERSION,
            provider,
            model,
            fields,
            dict(job.settings),
            [m.model_dump(mode="json") for m in job.covered],
        ]
        return f"{NAMESPACE}:ctxsum:v{KEY_VERSION}:{provider}:{model}:{digest(material)}"
    except (TypeError, ValueError):
        logger.debug("step %s: summary not cacheable (key material not JSON)", job.step)
        return ""


async def summarise(job: SummaryJob, context: StepContext) -> ContextSummary | None:
    """The summary of `job.covered`, from the cache or the model; None if the reply is invalid.

    Raises:
        ModelError: the summariser call failed for another reason.
    """
    ctx = run_context(context)
    cache = job.cache if job.cache is not None else context.cache  # opt-in only
    key = summary_key(job) if cache is not None else ""
    entry = SummaryEntry(ctx, next_entry_id(ctx), job.step, job.model.provider, job.model.model)
    if cache is not None and key:
        cached = await _load(cache, key, job.step)
        if cached is not None:
            entry.settle(Usage(), cache_hit=True)
            return cached
    request = LLMRequest(
        messages=(
            Message(role="system", content=job.system_prompt),
            Message(role="user", content=data_message(job.covered)),
        ),
        tools=(),
        tool_choice="none",
        output_schema=SUMMARY_SCHEMA,
        max_output_tokens=job.max_tokens,
        stage=STAGE,
        metadata={"call_id": entry.entry_id},  # an instrumented model reuses this entry id
    )
    try:
        with use_context(ctx):
            response = await job.model.generate(request)
    except (ModelOutputInvalid, ModelContentFiltered) as exc:
        entry.settle(Usage(), ok=False, error_kind=ErrorInfo.from_exception(exc).kind)
        log_failure(logger, logging.WARNING, exc, "step %s: no summary; messages kept", job.step)
        return None
    except Exception as exc:
        entry.settle(Usage(), ok=False, error_kind=ErrorInfo.from_exception(exc).kind)
        log_failure(logger, logging.WARNING, exc, "step %s: summariser call failed", job.step)
        raise
    context.add_summary_cost(entry.settle(response.usage, model=response.model))
    try:
        summary = response.parse(ContextSummary)
    except (ModelOutputInvalid, ModelContentFiltered) as exc:
        log_failure(logger, logging.WARNING, exc, "step %s: reply is not a summary; kept", job.step)
        return None
    if cache is not None and key:
        await _store(cache, key, summary, job)
    return summary


async def _load(cache: CachePort, key: str, step: str) -> ContextSummary | None:
    try:
        data = await cache.get(key)
    except Exception as exc:
        log_failure(logger, logging.WARNING, exc, "step %s: summary cache get failed", step)
        return None
    envelope = decode(data, _KIND) if data is not None else None
    if envelope is None:
        return None
    try:
        return ContextSummary.model_validate(envelope.payload)
    except ValidationError:
        logger.debug("step %s: cached summary is malformed; treated as a miss", step)
        return None


async def _store(cache: CachePort, key: str, summary: ContextSummary, job: SummaryJob) -> None:
    try:
        await cache.set(key, encode(_KIND, summary.model_dump(mode="json")), ttl=job.ttl)
    except Exception as exc:
        log_failure(logger, logging.WARNING, exc, "step %s: summary cache set failed", job.step)
