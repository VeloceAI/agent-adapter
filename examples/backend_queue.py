"""Queues: one `handle_envelope` for async consumers, sync job runners and Temporal activities."""

import base64
import functools
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from backend_common import Backend, build_backend

from agent_adapter import AdapterError, MemoryApprovalStore
from agent_adapter.handlers import handle_envelope
from agent_adapter.runtime import LoopRunner

logger = logging.getLogger(__name__)

# Stand-in: producers, consumers and the review app share one store (SQL, Redis, DynamoDB).
SHARED_STORE = MemoryApprovalStore()
TIMEOUT_S = 60


@functools.cache
def backend() -> Backend:
    """One per worker process, built on first use (after any fork)."""
    return build_backend(store=SHARED_STORE)


@dataclass
class Message:
    """One delivery: the broker's id (stable across redeliveries) and the raw body."""

    id: str
    body: bytes


@dataclass
class FakeQueue:
    """Stands in for SQS, Pub/Sub or Kafka: receive, ack, dead-letter."""

    pending: list[Message] = field(default_factory=list)
    acked: list[str] = field(default_factory=list)
    dead: list[str] = field(default_factory=list)

    async def receive(self) -> list[Message]:
        batch, self.pending = self.pending, []
        return batch

    async def ack(self, message: Message) -> None:
        self.acked.append(message.id)

    async def dead_letter(self, message: Message) -> None:
        self.dead.append(message.id)
        await self.ack(message)


# --8<-- [start:consumer]
async def consume_once(queue: FakeQueue, *, source: str = "kafka") -> list[dict[str, Any]]:
    """One poll of an async consumer (aiokafka, aiobotocore SQS, Pub/Sub, Arq, Taskiq)."""
    replies = []
    for message in await queue.receive():
        try:
            reply = await handle_envelope(
                backend().executor, message.body, message_id=message.id, source=source
            )
        except AdapterError as exc:
            if not exc.retryable:  # a bad message or a denied call: retrying cannot help
                await queue.dead_letter(message)
            continue  # retryable: no ack, so the broker redelivers it
        await queue.ack(message)  # ok and pending_approval are both done for this message
        replies.append(reply)
    return replies


# --8<-- [end:consumer]


# --8<-- [start:jobs]
RUNNER = LoopRunner()  # one warm loop per worker process, shared by every job


def tool_job(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A sync job body (RQ, Dramatiq, Huey): errors raise, so the runner records them."""
    return RUNNER.run(lambda: handle_envelope(backend().executor, payload), timeout=TIMEOUT_S)


async def tool_activity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A Temporal activity: never call the adapter from workflow code (it is replayed)."""
    return await handle_envelope(backend().executor, payload)


# --8<-- [end:jobs]


# --8<-- [start:pubsub]
async def pubsub_push(body: Mapping[str, Any]) -> dict[str, Any]:
    """A Pub/Sub push request body: base64 `data`, a stable `messageId`."""
    message = body["message"]
    data = base64.b64decode(message["data"])
    return await handle_envelope(
        backend().executor, data, message_id=message["messageId"], source="pubsub"
    )


# --8<-- [end:pubsub]
