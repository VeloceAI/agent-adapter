import asyncio
import json
from collections.abc import AsyncIterator

from agent_adapter import Registry
from agent_adapter.exec import Executor
from agent_adapter.voice import VoiceAgent, VoiceApproval
from agent_adapter.voice.socket import VoiceSocket, text_socket
from agent_adapter.voice.testing import settle
from tests.fakes.models import ScriptedModel, calls, text

registry = Registry()
refunds: list[str] = []


@registry.tool(effects="destructive", confirm=True)
async def refund(order: str) -> str:
    """Refund an order."""
    refunds.append(order)
    return "refunded"


class Client:
    """The browser side: messages it sends, and what it received."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[str | None] = asyncio.Queue()
        self.received: list[dict] = []

    def send(self, **message: object) -> None:
        self.inbox.put_nowait(json.dumps(message))

    def hang_up(self) -> None:
        self.inbox.put_nowait(None)

    async def _messages(self) -> AsyncIterator[str]:
        while (item := await self.inbox.get()) is not None:
            yield item

    async def _receive(self, text: str) -> None:
        self.received.append(json.loads(text))

    def socket(self):
        return text_socket(self._receive, self._messages())

    def says(self) -> list[tuple[int, str]]:
        return [(m["id"], m["text"]) for m in self.received if m["type"] == "say"]


def start(model, *, approval=None, timeout=5.0):
    client = Client()
    voice = VoiceSocket(client.socket(), played_timeout_s=timeout)
    agent = VoiceAgent(model, Executor(registry, approval=approval), voice.speaker)
    return client, agent, asyncio.create_task(voice.run(agent))


async def test_history_keeps_only_lines_the_client_says_it_played():
    client, agent, run = start(ScriptedModel([text("First line. Second line. Third line.")]))
    client.send(type="transcript", text="hel", final=False)
    client.send(type="transcript", text="hello", final=True)
    await settle()
    assert client.says() == [(1, "First line.")]  # the next line waits for `played`
    client.send(type="played", id=1)
    await settle()
    assert client.says()[-1] == (2, "Second line.")
    client.send(type="speech_started")
    await settle()
    assert client.received[-1] == {"type": "stop"}
    assert agent.history[-1].content == "First line."
    client.hang_up()
    await asyncio.wait_for(run, 1)


async def test_a_spoken_yes_over_the_socket_approves_and_junk_is_ignored():
    refunds.clear()
    model = ScriptedModel([calls(("refund", {"order": "A1"})), text("Refunded.")])
    client, _, run = start(model, approval=VoiceApproval(timeout_s=5))
    for junk in ("not json", "[1, 2]", json.dumps({"type": "dance"}), "x" * 70_000):
        client.inbox.put_nowait(junk)
    client.send(type="transcript", text="refund A1", final=True)
    await settle()
    assert client.says() == [(1, "Shall I run refund? Say yes to confirm.")]
    client.send(type="played", id=1)
    client.send(type="played", id="1")  # ignored: not an int
    client.send(type="transcript", text="yes", final=True)
    await settle()
    assert refunds == ["A1"]
    client.send(type="hangup")
    await asyncio.wait_for(run, 1)


async def test_a_client_that_never_acknowledges_is_trusted_after_the_timeout():
    client, agent, run = start(ScriptedModel([text("Only line.")]), timeout=0.01)
    client.send(type="transcript", text="hi", final=True)
    await asyncio.sleep(0.05)
    await agent.wait()
    assert agent.history[-1].content == "Only line."
    client.hang_up()
    await asyncio.wait_for(run, 1)


async def test_a_stop_to_a_closed_client_is_not_an_error():
    async def closed(_: str) -> None:
        raise ConnectionError("gone")

    async def nothing() -> AsyncIterator[str]:
        return
        yield

    voice = VoiceSocket(text_socket(closed, nothing()))
    await voice.speaker.stop()
