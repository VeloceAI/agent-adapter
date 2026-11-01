"""`tool_handler`: a sync `handler(event, context)` for AWS Lambda and other function hosts."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from typing import Any, Protocol, TypeVar

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.exec._sugar import shared_runner
from agent_adapter.exec.executor import Executor
from agent_adapter.handlers.consume import Reply, handle_envelope
from agent_adapter.runtime.sync import _positive_number, run_sync

logger = logging.getLogger(__name__)

T = TypeVar("T")
Handler = Callable[..., dict[str, Any]]
SQS_SOURCE = "aws:sqs"
MIN_BUDGET_S = 0.001


class Runner(Protocol):
    """Runs async work from sync code: a shared `LoopRunner` by default, or `RunSync`."""

    def run(self, work: Callable[[], Awaitable[T]], *, timeout: float | None = None) -> T:
        """Run `work()` to completion; `TimeoutError` past `timeout` seconds."""
        ...


class RunSync:
    """A `Runner` with one fresh event loop per invocation (`run_sync`); nothing stays warm."""

    def run(self, work: Callable[[], Awaitable[T]], *, timeout: float | None = None) -> T:
        """Run `work()` on a fresh loop."""
        return run_sync(work, timeout=timeout)


def tool_handler(
    executor: Executor,
    *,
    tools: Collection[str] | None = None,
    timeout: float = 60.0,
    margin_s: float = 1.0,
    runner: Runner | None = None,
) -> Handler:
    """A sync `handler(event, context=None)` that runs envelopes through `executor`.

    Events: an SQS batch (`Records`, returns `{"batchItemFailures": [...]}`), an
    EventBridge event (the envelope in `detail`), or the envelope itself (direct invoke,
    other hosts). Each invocation gets `min(timeout, remaining time - margin_s)` seconds
    when `context` has `get_remaining_time_in_millis()`. Without `runner`, handlers share
    the process's sync loop (the one `run_sync` uses), so warm invocations reuse
    `LoopLocal` clients. A direct invoke takes its delivery id from `context.aws_request_id`,
    which Lambda keeps across async retries.

    Raises:
        ToolUsageError: `timeout` or `margin_s` is not a positive (or zero margin) number.
    """
    if not _positive_number(timeout):
        raise ToolUsageError(f"tool_handler timeout must be a positive number, got {timeout!r}")
    if not (_positive_number(margin_s) or margin_s == 0):
        raise ToolUsageError(f"tool_handler margin_s must be >= 0, got {margin_s!r}")
    run = runner if runner is not None else shared_runner()

    def handler(event: Mapping[str, Any], context: Any = None) -> dict[str, Any]:
        """Handle one invocation; errors of a single event raise so the host retries it."""
        budget = _budget(timeout, margin_s, context)
        if not isinstance(event, Mapping):
            raise ToolUsageError(f"event must be a JSON object, got {type(event).__name__}")
        if "Records" in event:
            return _sqs_batch(executor, event["Records"], tools, run, budget)
        payload, message_id, source = _single(event, context)

        def work() -> Awaitable[Reply]:
            return handle_envelope(
                executor, payload, tools=tools, message_id=message_id, source=source
            )

        try:
            return run.run(work, timeout=budget)
        except TimeoutError:
            logger.warning("event ran out of time after %.3gs (message %r)", budget, message_id)
            raise

    return handler


def _single(event: Mapping[str, Any], context: Any) -> tuple[Any, str, str]:
    """(envelope payload, delivery id, source) of a non-batch event."""
    if "detail-type" in event and "detail" in event:  # EventBridge: `id` is stable on retry
        event_id = event.get("id")
        return event["detail"], event_id if isinstance(event_id, str) else "", "eventbridge"
    request_id = getattr(context, "aws_request_id", None)  # stable across async retries
    if isinstance(request_id, str) and request_id:
        return event, request_id, "lambda"
    return event, "", ""


def _sqs_batch(
    executor: Executor,
    records: Any,
    tools: Collection[str] | None,
    run: Runner,
    budget: float,
) -> dict[str, Any]:
    """Run each message in order; failed and unreached messages go back to the queue."""
    checked = _sqs_records(records)
    fifo = any(_is_fifo(record) for record in checked)
    done: set[str] = set()

    async def work() -> None:
        for record in checked:
            message_id = record["messageId"]
            try:
                await handle_envelope(
                    executor, record["body"], tools=tools, message_id=message_id, source="sqs"
                )
            except Exception:  # logged by handle_envelope; reported below as a batch failure
                if fifo:  # later messages of a FIFO group must not overtake this one
                    return
                continue
            done.add(message_id)

    try:
        run.run(work, timeout=budget)
    except TimeoutError:
        left = len(checked) - len(done)
        logger.warning("sqs batch ran out of time after %.3gs: %d messages left", budget, left)
    failures = [{"itemIdentifier": r["messageId"]} for r in checked if r["messageId"] not in done]
    if failures:
        logger.warning("sqs batch: %d of %d messages failed", len(failures), len(checked))
    return {"batchItemFailures": failures}


def _sqs_records(records: Any) -> list[Mapping[str, Any]]:
    """The records, checked before anything runs: SQS only, each with an id and a body."""
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ToolUsageError("event 'Records' must be a list")
    checked: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping) or record.get("eventSource") != SQS_SOURCE:
            raise ToolUsageError("tool_handler takes SQS batches only (eventSource 'aws:sqs')")
        if not isinstance(record.get("messageId"), str) or not record["messageId"]:
            raise ToolUsageError("an SQS record has no messageId")
        if not isinstance(record.get("body"), str):
            raise ToolUsageError("an SQS record has no string body")
        checked.append(record)
    return checked


def _is_fifo(record: Mapping[str, Any]) -> bool:
    arn = record.get("eventSourceARN")
    attributes = record.get("attributes")
    grouped = isinstance(attributes, Mapping) and "MessageGroupId" in attributes
    return grouped or (isinstance(arn, str) and arn.endswith(".fifo"))


def _budget(timeout: float, margin_s: float, context: Any) -> float:
    """Seconds this invocation may run: `timeout`, capped by the host's remaining time."""
    remaining = getattr(context, "get_remaining_time_in_millis", None)
    if not callable(remaining):
        return timeout
    left = float(remaining()) / 1000 - margin_s
    return max(min(timeout, left), MIN_BUDGET_S)
