from array import array
from types import SimpleNamespace

from agent_adapter.voice.quality import InputCheck, Issue, OutputCheck, measure
from agent_adapter.voice.testing import silence, tone
from tests.fakes.sinks import RecordingSink


def pcm(values: list[int]) -> bytes:
    return array("h", values).tobytes()


def fake_session(out_rate: int = 16_000):
    said: list[str] = []
    sink = RecordingSink()
    agent = SimpleNamespace(
        say=lambda text, record=True: said.append(text),
        executor=SimpleNamespace(sink=sink),
        ctx=SimpleNamespace(run_id="run-1", call_id=None, thread_id=None, parent_call_id=None),
    )
    return SimpleNamespace(agent=agent, transport=SimpleNamespace(out_rate=out_rate)), said, sink


def test_measure_level_clipping_jumps_and_zero_runs():
    quiet = measure(silence(20))
    assert quiet.level_db == -120.0 and quiet.zero_run == 320
    loud = measure(pcm([32_767, -32_768, 0, 0, 0, 100]))
    assert loud.clipped == 2 and loud.peak == 1.0 and loud.zero_run == 3
    assert round(loud.jump, 2) == 2.0
    assert measure(pcm([100, 200]), previous=-30_000).jump > 0.9
    assert measure(b"").samples == 0
    assert measure(pcm([1, 256, 0, 1])).zero_run == 1  # bytes 01 00 00 01: no zero run across
    assert round(measure(tone(20, level=16_384)).level_db) == -6  # a square wave


def test_gate_silences_hiss_and_holds_open_after_speech():
    check = InputCheck(gate_db=-40, hold_ms=40)
    hiss = tone(20, level=50)
    assert check(hiss) == bytes(len(hiss))
    speech = tone(20, level=8_000)
    assert check(speech) == speech
    assert check(hiss) == hiss and check(hiss) == hiss  # 40 ms of hold
    assert check(hiss) == bytes(len(hiss))
    assert InputCheck()(hiss) == hiss  # no gate_db: measures only


def turn_of(check: InputCheck, *frames: bytes) -> list[str]:
    for frame in frames:
        check(frame)
    return check.judge().problems


def test_clipping_dropouts_quiet_and_noisy_turns_are_named():
    floor = [tone(20, level=30)] * 5
    assert turn_of(InputCheck(), *floor, *[tone(20, level=8_000)] * 10) == []
    clipped = pcm([32_767, -32_768] * 160)
    assert turn_of(InputCheck(), *floor, *[clipped] * 10) == ["clipping"]
    gap = tone(10, level=8_000) + silence(10)
    assert turn_of(InputCheck(), *floor, *[gap] * 15) == ["dropouts"]
    assert turn_of(InputCheck(), *floor, *[tone(20, level=180)] * 10) == ["too quiet"]
    noisy = InputCheck()
    assert turn_of(noisy, *[tone(20, level=1_000)] * 5, *[tone(20, level=8_000)] * 10) == ["noisy"]
    assert noisy.counts["noisy"] == 1 and noisy.last_turn is not None


def test_a_bad_turn_asks_the_caller_to_repeat_once_then_lets_through():
    session, said, _ = fake_session()
    check = InputCheck(max_repeats=1)
    check.started(session)
    clipped = pcm([32_767, -32_768] * 160)
    for frame in [tone(20, level=30)] * 5 + [clipped] * 10:
        check.caller_audio(frame)  # used only as an observer: it measures here
    assert check.caller_said("refund four thousand") is True
    assert said == ["Sorry, I didn't catch that clearly. Could you say it again?"]
    for frame in [clipped] * 10:
        check.caller_audio(frame)
    assert check.caller_said("refund four thousand") is None  # asked once already
    assert check.caller_said("fine now") is None and check.repeats == 0


def test_measuring_only_and_no_agent_never_keep_the_words():
    clipped = pcm([32_767, -32_768] * 160)
    quiet = InputCheck(ask_again=False)
    quiet(clipped)
    assert quiet.caller_said("hello") is None and quiet.counts["clipping"] == 1
    alone = InputCheck()
    alone(clipped)
    assert alone.caller_said("hello") is None


async def test_input_counts_go_to_the_trace_at_the_end():
    session, _, sink = fake_session()
    check = InputCheck(ask_again=False)
    check.started(session)
    check(tone(20, level=30))
    check(pcm([32_767, -32_768] * 160))
    check.caller_said("hi")
    await check.ended()
    [event] = [e for e in sink.events if e.type == "custom.voice_quality"]
    assert event.data == {"side": "caller", "clipping": 1}


async def test_output_check_finds_clicks_clipping_and_dead_air():
    session, _, sink = fake_session(out_rate=16_000)
    seen: list[Issue] = []
    check = OutputCheck(on_issue=seen.append, silence_s=0.1)
    check.started(session)
    assert check.rate == 16_000
    assert check.feed(tone(20)) == []
    assert [i.kind for i in check.feed(pcm([0, 30_000, 0] * 10))] == ["click"]
    kinds = [i.kind for frame in [silence(20)] * 8 for i in check.feed(frame)]
    assert kinds == ["silence"]  # flagged once per stretch
    ramp = list(range(0, 32_767, 1_000))
    assert [i.kind for i in check.feed(pcm(ramp + [32_767] * 50))] == ["clipping"]
    check.agent_audio(pcm([31_000] * 10))  # the observer port feeds the same check
    assert check.feed(b"") == []
    assert [i.kind for i in seen] == ["click", "silence", "clipping"]
    assert dict(check.counts) == {"click": 1, "clipping": 1, "silence": 1}
    await check.ended()
    [event] = [e for e in sink.events if e.type == "custom.voice_quality"]
    assert event.data == {"side": "agent", "click": 1, "clipping": 1, "silence": 1}


def test_output_seam_click_and_a_failing_callback():
    def broken(issue: Issue) -> None:
        raise RuntimeError("boom")

    check = OutputCheck(rate=16_000, on_issue=broken)
    check.feed(pcm([30_000] * 10))
    assert [i.kind for i in check.feed(pcm([-30_000] * 10))] == ["click"]  # across chunks
    assert check.counts["click"] == 1
