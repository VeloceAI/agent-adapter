"""VoiceApproval with many calls on one executor: each question and answer stays on its call."""

import asyncio

from agent_adapter import Registry
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.voice import VoiceAgent, VoiceApproval
from agent_adapter.voice.approval import default_prompt
from agent_adapter.voice.testing import RecordingSpeaker, settle
from tests.fakes.models import ScriptedModel, calls, text


def _request(run_id: str, tool: str = "refund", **arguments: object) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=f"a-{run_id}", run_id=run_id, call_id="c", tool=tool, arguments=arguments
    )


class Line:
    """One caller's speaker as the approval sees it: records each prompt."""

    def __init__(self) -> None:
        self.said: list[str] = []

    async def say(self, text: str) -> None:
        self.said.append(text)


async def _armed(approval: VoiceApproval, run_id: str) -> int:
    for _ in range(100):
        question = approval.armed_question(run_id)
        if question is not None:
            return question
        await asyncio.sleep(0)
    raise AssertionError("the question never finished playing")


async def test_a_prompt_goes_only_to_the_caller_whose_run_asked_and_only_they_can_answer():
    approval = VoiceApproval(timeout_s=0.2)
    a, b = Line(), Line()
    approval.bind(a.say, run_id="A")
    approval.bind(b.say, run_id="B")
    asked = asyncio.create_task(approval.request(_request("A", amount=9999)))
    question = await _armed(approval, "A")
    assert a.said and b.said == []
    assert not approval.answer("yes", run_id="B", question=question)  # B has no question
    assert approval.waiting_for("A") and not approval.waiting_for("B")
    decision = await asked
    assert not decision.approved and decision.reason == "no answer"


async def test_the_callers_own_yes_after_the_question_approves():
    approval = VoiceApproval(timeout_s=1)
    line = Line()
    approval.bind(line.say, run_id="A")
    asked = asyncio.create_task(approval.request(_request("A")))
    question = await _armed(approval, "A")
    assert approval.answer("yes", run_id="A", question=question)
    assert (await asked).approved


async def test_a_run_with_no_bound_call_is_denied_without_asking_anyone():
    approval = VoiceApproval(timeout_s=1)
    line = Line()
    approval.bind(line.say, run_id="A")
    decision = await approval.request(_request("stranger"))
    assert not decision.approved and line.said == []


async def test_one_yes_approves_only_one_of_two_concurrent_questions():
    approval = VoiceApproval(timeout_s=0.2)
    line = Line()
    approval.bind(line.say, run_id="A")
    first = asyncio.create_task(approval.request(_request("A", "transfer", amount=5000)))
    second = asyncio.create_task(approval.request(_request("A", "delete_account", user="bob")))
    question = await _armed(approval, "A")
    assert len(line.said) == 1  # the second question waits for the first to be decided
    assert approval.answer("yes", run_id="A", question=question)
    assert (await first).approved
    assert not (await second).approved
    assert len(line.said) == 2 and "delete account" in line.said[1]


async def test_words_spoken_before_or_during_the_question_do_not_answer_it():
    approval = VoiceApproval(timeout_s=0.2)
    playing = asyncio.Event()

    async def slow_say(text: str) -> None:
        await playing.wait()

    approval.bind(slow_say, run_id="A")
    asked = asyncio.create_task(approval.request(_request("A")))
    await settle()
    assert approval.waiting_for("A") and approval.armed_question("A") is None
    assert approval.answer("yes", run_id="A", question=None)  # taken, but too early
    playing.set()
    question = await _armed(approval, "A")
    assert approval.answer("yes", run_id="A", question=None)  # began before the question ended
    assert approval.answer("yes", run_id="A", question=question - 1)  # an older question
    decision = await asked
    assert not decision.approved and decision.reason == "no answer"


async def test_unbinding_a_call_denies_its_open_question_at_once():
    approval = VoiceApproval(timeout_s=30)
    line = Line()
    approval.bind(line.say, run_id="A")
    asked = asyncio.create_task(approval.request(_request("A")))
    await _armed(approval, "A")
    approval.unbind("A")
    decision = await asyncio.wait_for(asked, 1)
    assert not decision.approved and not approval.waiting_for("A")


async def test_a_run_id_can_be_bound_to_only_one_call():
    from agent_adapter.errors import ConfigError

    approval = VoiceApproval()
    approval.bind(Line().say, run_id="A")
    try:
        approval.bind(Line().say, run_id="A")
    except ConfigError:
        return
    raise AssertionError("a second call took over run A")


def test_the_prompt_says_the_arguments_in_plain_words_and_keeps_long_values_short():
    assert default_prompt(_request("A", amount=9999)) == (
        "Shall I run refund with amount 9999? Say yes to confirm."
    )
    two = default_prompt(_request("A", "send_money", to_account="Bob", urgent=True))
    assert two == "Shall I run send money with to account Bob and urgent yes? Say yes to confirm."
    long = default_prompt(_request("A", note="x " * 500, items=list(range(1000))))
    assert len(long) < 260 and "..." in long
    hinted = ApprovalRequest(approval_id="a", run_id="A", call_id="c", tool="t", hint="book 4A")
    assert default_prompt(hinted) == "Shall I book 4A? Say yes to confirm."


async def test_a_host_can_give_one_tool_its_own_prompt():
    approval = VoiceApproval(
        timeout_s=0.01, prompts={"refund": lambda r: f"Refund {r.arguments['amount']} dollars?"}
    )
    line = Line()
    approval.bind(line.say, run_id="A")
    await approval.request(_request("A", amount=5))
    await approval.request(_request("A", "other", amount=5))
    assert line.said == [
        "Refund 5 dollars?",
        "Shall I run other with amount 5? Say yes to confirm.",
    ]


registry = Registry()
refunds: list[int] = []


@registry.tool(effects="destructive", confirm=True)
async def refund(amount: int) -> str:
    """Refund."""
    refunds.append(amount)
    return "refunded"


async def test_two_voice_agents_on_one_executor_keep_their_questions_apart():
    refunds.clear()
    executor = Executor(registry, approval=VoiceApproval(timeout_s=0.3))
    speaker_a, speaker_b = RecordingSpeaker(), RecordingSpeaker()
    agent_a = VoiceAgent(
        ScriptedModel([calls(("refund", {"amount": 9999})), text("Not done.")]),
        executor,
        speaker_a,
    )
    agent_b = VoiceAgent(ScriptedModel([text("Hello.")]), executor, speaker_b)
    await agent_a.user_said("refund me")
    await settle()
    assert speaker_a.said == ["Shall I run refund with amount 9999? Say yes to confirm."]
    assert speaker_b.said == []
    await agent_b.user_started_speaking()
    await agent_b.user_said("yes")  # caller B's yes is B's own new turn
    await agent_a.wait()
    await agent_b.wait()
    assert refunds == [] and speaker_b.said == ["Hello."]
    await agent_a.close()
    await agent_b.close()
    assert not executor.approval.waiting  # type: ignore[union-attr]


async def test_a_voice_agent_ignores_a_yes_spoken_over_its_question():
    refunds.clear()
    executor = Executor(registry, approval=VoiceApproval(timeout_s=0.2))
    speaker = RecordingSpeaker(hold=True)
    model = ScriptedModel([calls(("refund", {"amount": 5})), text("Not done.")])
    agent = VoiceAgent(model, executor, speaker)
    await agent.user_said("refund me")
    await settle()
    assert speaker.started == ["Shall I run refund with amount 5? Say yes to confirm."]
    await agent.user_started_speaking()  # the caller talks over the question
    await agent.user_said("yes")
    speaker.release(2)
    await agent.wait()
    assert refunds == []
