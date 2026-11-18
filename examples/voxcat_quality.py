"""VoxCat quality checks offline: spoken money and dates, a misheard name fixed, an unsure
amount read back before a refund, and bad audio caught on both sides of the call."""

import asyncio
import math
from array import array

from agent_adapter import ApprovalRequest
from agent_adapter.voice import VoiceApproval
from agent_adapter.voice.quality import (
    InputCheck,
    KnownNames,
    OutputCheck,
    SpokenForm,
    UnsureWords,
    Word,
    check_stt,
)
from agent_adapter.voice.testing import ScriptedSTT


def wave(ms: float, level: int, rate: int = 16_000) -> bytes:
    count = rate * ms // 1000
    return array("h", (int(level * math.sin(i / 7)) for i in range(int(count)))).tobytes()


async def main() -> None:
    # --8<-- [start:spoken]
    spoken = SpokenForm(locale="en-IN", pronounce={"VoxCat": "vox cat"})
    print(spoken("VoxCat refunds ₹4,50,000 on 18/11/2026 at 14:30."))
    print(SpokenForm()("Dr. Rao paid $1,249.99, approx. 3.5% more, on 2026-11-18."))
    # --8<-- [end:spoken]

    # --8<-- [start:names]
    names = KnownNames(["Rahul", "Priyanka", "Acme Corp"])
    print(names("This is Rahool from Acmy Corp."))
    print([(fix.heard, fix.name) for fix in names.last_fixes])
    # --8<-- [end:names]

    # --8<-- [start:unsure]
    unsure = UnsureWords(threshold=0.85)
    approval = VoiceApproval(readback=unsure.readback, timeout_s=0.1)
    # what a hosted speech-to-text heard, with its confidence in each word
    unsure.hear([Word("refund", 0.98), Word("4,500", 0.41), Word("please", 0.97)], run_id="call-1")
    refund = ApprovalRequest(
        approval_id="a1", run_id="call-1", call_id="c1", tool="refund", arguments={"amount": 4500}
    )

    async def say(line: str) -> None:  # a VoiceAgent binds its speaker here
        print(line)

    approval.bind(say, run_id="call-1")
    decision = await approval.request(refund)  # nobody answers in this script
    print("approved" if decision.approved else f"denied: {decision.reason}")
    # --8<-- [end:unsure]

    # --8<-- [start:stt]
    stt = check_stt(ScriptedSTT(["pay priyonka"]), fixes=[names], unsure=unsure)
    print(await stt.transcribe(b"", 16_000))
    # VoiceSession(transport, stt=stt, tts=tts, turns=turns, observers=[stt, ...])
    # --8<-- [end:stt]

    # --8<-- [start:audio]
    caller = InputCheck(gate_db=-55)  # VoiceSession(filters=[caller], observers=[caller])
    for _ in range(10):
        caller(wave(20, level=30))  # the line's own hiss: the noise floor
    for _ in range(10):
        caller(array("h", [32_767, -32_768] * 160).tobytes())  # a shout, clipped
    print(caller.judge().problems)

    agent = OutputCheck(rate=16_000)  # VoiceSession(observers=[agent])
    agent.feed(wave(20, level=8_000))
    print([issue.kind for issue in agent.feed(array("h", [0, 30_000, 0] * 50).tobytes())])
    # --8<-- [end:audio]


if __name__ == "__main__":
    asyncio.run(main())
