from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.ports import (
    ApprovalPort,
    CachePort,
    EventSink,
    LedgerPort,
    LLMPort,
    PromptCachePlugin,
)
from agent_adapter.core.types import ApprovalDecision
from tests.fakes.cache import BrokenCache, DictCache
from tests.fakes.sinks import RecordingLedger, RecordingSink


class Approver:
    async def request(self, request):
        return ApprovalDecision(approved=True)


class Model:
    provider = "fake"
    model = "m"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(provider="fake", model="m", message=Message(role="assistant"))

    async def stream(self, request):
        yield await self.generate(request)

    def cache_key_fields(self):
        return {}


class Plugin:
    name = "p"
    providers = frozenset({"fake"})

    def key_fields(self):
        return {}

    async def apply(self, request, kwargs, *, model, client):
        return kwargs


def test_fakes_satisfy_the_ports():
    assert isinstance(DictCache(), CachePort)
    assert isinstance(BrokenCache(), CachePort)
    assert isinstance(RecordingLedger(), LedgerPort)
    assert isinstance(RecordingSink(), EventSink)
    assert isinstance(Approver(), ApprovalPort)
    assert isinstance(Model(), LLMPort)
    assert isinstance(Plugin(), PromptCachePlugin)


def test_objects_missing_methods_do_not_satisfy_ports():
    assert not isinstance(object(), CachePort)
    assert not isinstance(RecordingSink(), LedgerPort)
