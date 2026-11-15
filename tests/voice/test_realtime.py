import asyncio
import json
from collections.abc import AsyncIterator

from agent_adapter import Registry
from agent_adapter.cost import Ledger, Prices
from agent_adapter.exec import Executor, progress
from agent_adapter.voice import VoiceApproval
from agent_adapter.voice.gemini_live import GeminiLive
from agent_adapter.voice.openai_realtime import OpenAIRealtime
from agent_adapter.voice.realtime import RealtimeAgent
from agent_adapter.voice.socket import text_socket
from agent_adapter.voice.testing import settle

registry = Registry()
done: list[str] = []
gate = asyncio.Event()


@registry.tool(effects="destructive")
async def book(seat: str) -> str:
    """Book a seat."""
    progress("One moment")
    await gate.wait()
    done.append(seat)
    return f"booking #{seat}"


@registry.tool(effects="read_only")
async def find(seat: str) -> str:
    """Look a seat up."""
    await gate.wait()
    done.append(f"found {seat}")
    return "free"


@registry.tool(effects="destructive", confirm=True)
async def refund(order: str) -> str:
    """Refund an order."""
    done.append(f"refunded {order}")
    return "refunded"


class Provider:
    """The model's side of the socket: what it sends, and what it received."""

    def __init__(self) -> None:
        self.outbox: asyncio.Queue[str | None] = asyncio.Queue()
        self.received: list[dict] = []

    def send(self, message: object) -> None:
        self.outbox.put_nowait(message if isinstance(message, str) else json.dumps(message))

    async def _messages(self) -> AsyncIterator[str]:
        while (item := await self.outbox.get()) is not None:
            yield item

    async def _receive(self, text: str) -> None:
        self.received.append(json.loads(text))

    def close(self) -> None:
        self.outbox.put_nowait(None)


def start(codec, *, approval=None, ledger=None, minute_item=None):
    global gate
    done.clear()
    gate = asyncio.Event()
    provider = Provider()
    executor = Executor(registry, approval=approval, ledger=ledger)
    agent = RealtimeAgent(
        codec,
        text_socket(provider._receive, provider._messages()),
        executor,
        instructions="Help with seats.",
        minute_item=minute_item,
    )
    return provider, agent, asyncio.create_task(agent.run())


def openai_call(call_id: str, name: str, args: str) -> dict:
    return {"type": "response.function_call_arguments.done", "call_id": call_id, "name": name,
            "arguments": args}  # fmt: skip


async def test_openai_runs_a_call_and_returns_its_output_then_asks_for_a_reply():
    provider, _, run = start(OpenAIRealtime())
    provider.send(openai_call("c1", "book", '{"seat": "4A"}'))
    await settle()
    gate.set()
    await settle()
    setup, spoken, output, reply = provider.received
    assert setup["type"] == "session.update"
    assert [t["name"] for t in setup["session"]["tools"]] == ["book", "find", "refund"]
    assert spoken["response"]["instructions"].endswith("One moment")
    assert output["item"] == {"type": "function_call_output", "call_id": "c1",
                              "output": "booking #4A"}  # fmt: skip
    assert reply == {"type": "response.create"}
    provider.close()
    await asyncio.wait_for(run, 1)


async def test_openai_does_not_talk_over_a_caller_who_cut_in_and_rejects_bad_json():
    provider, _, run = start(OpenAIRealtime(session={"tool_choice": "required"}))
    provider.send(openai_call("c1", "book", '{"seat": "4A"}'))
    await settle()
    provider.send({"type": "input_audio_buffer.speech_started"})
    provider.send({"type": "error", "error": {"message": "x"}})
    provider.send(openai_call("c2", "find", "{nope"))
    provider.send(openai_call("c3", "find", "[1]"))
    await settle()
    gate.set()
    await settle()
    assert provider.received[0]["session"]["tool_choice"] == "required"
    sent = [m for m in provider.received[1:] if m["type"] != "response.create" or "response" in m]
    outputs = {m["item"]["call_id"]: m["item"]["output"] for m in sent if "item" in m}
    assert "not JSON" in outputs["c2"] and "not JSON" in outputs["c3"]
    assert outputs["c1"] == "booking #4A"
    at = next(
        i for i, m in enumerate(provider.received) if m.get("item", {}).get("call_id") == "c1"
    )
    assert provider.received[at + 1 :] == []  # c1 began before the cut: the caller has the floor
    provider.close()
    await asyncio.wait_for(run, 1)


async def test_openai_approval_needs_the_callers_own_yes():
    provider, _, run = start(OpenAIRealtime(), approval=VoiceApproval(timeout_s=5))
    provider.send(openai_call("c1", "refund", '{"order": "A1"}'))
    await settle()
    assert "Shall I run refund?" in provider.received[-1]["response"]["instructions"]
    provider.send({"type": "conversation.item.input_audio_transcription.completed",
                   "transcript": "yeah no"})  # fmt: skip
    await settle()
    assert done == []
    provider.send(openai_call("c2", "refund", '{"order": "A1"}'))
    await settle()
    provider.send({"type": "conversation.item.input_audio_transcription.completed",
                   "transcript": " "})  # fmt: skip
    provider.send({"type": "conversation.item.input_audio_transcription.completed",
                   "transcript": "yes"})  # fmt: skip
    await settle()
    assert done == ["refunded A1"]
    provider.close()
    await asyncio.wait_for(run, 1)


async def test_gemini_cancellation_stops_a_lookup_but_a_booking_finishes_and_is_reported():
    provider, _, run = start(GeminiLive("gemini-3.8-live"))
    provider.send({"setupComplete": {}})
    provider.send({"toolCall": {"functionCalls": [
        {"id": "b1", "name": "book", "args": {"seat": "4A"}},
        {"id": "f1", "name": "find", "args": {"seat": "4A"}},
    ]}})  # fmt: skip
    await settle()
    provider.send({"serverContent": {"interrupted": True}})
    provider.send({"toolCallCancellation": {"ids": ["b1", "f1", "gone"]}})
    await settle()
    gate.set()
    await settle()
    assert done == ["4A"]  # the lookup stopped; the booking finished once
    notes = [m["clientContent"] for m in provider.received if "clientContent" in m]
    assert notes[-1]["turnComplete"] is False
    assert "Already done: book finished: booking #4A" in notes[-1]["turns"][0]["parts"][0]["text"]
    assert not any("toolResponse" in m for m in provider.received)
    provider.send({"toolCall": {"functionCalls": [{"id": "b2", "name": "book",
                                                   "args": {"seat": "4A"}}]}})  # fmt: skip
    await settle()
    reply = provider.received[-1]["toolResponse"]["functionResponses"][0]
    assert reply == {"id": "b2", "name": "book", "response": {"result": "booking #4A"}}
    assert done == ["4A"]  # the re-issued call returned the first result
    provider.close()
    await asyncio.wait_for(run, 1)


async def test_gemini_joins_the_callers_words_and_approves_on_a_clear_yes():
    provider, _, run = start(GeminiLive("models/x", setup={"proactivity": {}}),
                             approval=VoiceApproval(timeout_s=5))  # fmt: skip
    await settle()
    setup = provider.received[0]["setup"]
    assert setup["model"] == "models/x" and "proactivity" in setup
    assert setup["systemInstruction"]["parts"][0]["text"] == "Help with seats."
    provider.send({"toolCall": {"functionCalls": [
        {"id": "r1", "name": "refund", "args": {"order": "A1"}}, "junk",
        {"id": "r0", "name": "refund", "args": "not a mapping"},
    ]}})  # fmt: skip
    await settle()
    prompts = [m["clientContent"] for m in provider.received if "clientContent" in m]
    assert "Shall I run refund?" in prompts[0]["turns"][0]["parts"][0]["text"]
    errors = [m["toolResponse"] for m in provider.received if "toolResponse" in m]
    assert "not JSON" in errors[0]["functionResponses"][0]["response"]["result"]
    provider.send({"serverContent": {"inputTranscription": {"text": "Ye"}}})
    provider.send({"serverContent": {"inputTranscription": {"text": "s."}}})
    await settle()
    assert done == []  # not final until the model replies
    provider.send({"serverContent": {"modelTurn": {"parts": []}}})
    await settle()
    assert done == ["refunded A1"]
    provider.close()
    await asyncio.wait_for(run, 1)


async def test_the_call_minutes_are_billed_and_junk_messages_are_ignored():
    units = {"rt:minute": {"provider": "rt", "per_unit": 0.06}}
    ledger = Ledger(prices=Prices.default().merged({"version": "t", "units": units}))
    provider, agent, run = start(GeminiLive("m"), ledger=ledger, minute_item="rt:minute")
    for junk in ("not json", "[1]", b'{"setupComplete": {}}'):
        provider.send(junk if isinstance(junk, str) else junk.decode())
    provider.send({"serverContent": {"turnComplete": True}})
    provider.close()
    await asyncio.wait_for(run, 1)
    assert [e.stage for e in ledger.entries(agent.ctx.run_id)] == ["rt:minute"]
