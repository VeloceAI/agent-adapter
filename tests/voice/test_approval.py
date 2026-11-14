import pytest

from agent_adapter.core.types import ApprovalRequest
from agent_adapter.voice import CallableSpeaker, VoiceApproval, is_yes


@pytest.mark.parametrize(
    "heard",
    [
        "yes",
        "Yes, please.",
        "yeah go ahead",
        "That" + chr(0x2019) + "s right",
        "OK do it",
        "confirm",
    ],
)
def test_clear_yes(heard):
    assert is_yes(heard)


@pytest.mark.parametrize(
    "heard", ["", "no", "yeah no", "yes but wait", "not yet", "the user agreed", "maybe", "?"]
)
def test_anything_else_is_no(heard):
    assert not is_yes(heard)


def _request(hint: str = "") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="a1", run_id="r", call_id="c", tool="refund_ticket", hint=hint
    )


async def test_an_unbound_port_denies():
    decision = await VoiceApproval().request(_request())
    assert not decision.approved and decision.decided_by == "voice"


async def test_the_prompt_uses_the_hint_and_an_answer_nobody_waits_for_is_refused():
    said: list[str] = []
    approval = VoiceApproval(timeout_s=0.01)

    async def say(text: str) -> None:
        said.append(text)

    approval.bind(say)
    assert not approval.answer("yes")
    decision = await approval.request(_request("refund the dinosaur ticket"))
    assert said == ["Shall I refund the dinosaur ticket? Say yes to confirm."]
    assert not decision.approved and decision.reason == "no answer"


async def test_callable_speaker_takes_sync_and_async_functions():
    said: list[str] = []
    stopped: list[bool] = []

    async def say(text: str) -> None:
        said.append(text)

    speaker = CallableSpeaker(say, lambda: stopped.append(True))
    await speaker.say("hi")
    await speaker.stop()
    await CallableSpeaker(said.append).stop()  # no stop function: a no-op
    assert said == ["hi"] and stopped == [True]
