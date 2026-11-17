"""Each quality check stays under 1 ms a frame or a sentence (median), with no benchmark plugin."""

import math
import statistics
import time
from array import array
from collections.abc import Callable
from typing import Any

import pytest

from agent_adapter.core.types import ApprovalRequest
from agent_adapter.voice.quality import (
    InputCheck,
    KnownNames,
    OutputCheck,
    SpokenForm,
    UnsureWords,
    Word,
)

BUDGET_MS = 1.0
SENTENCES = [
    "Pay ₹4,50,000 to Rahool at Acmy Corp by 18/11/2026 at 14:30, approx. 3.5% fee.",
    "Dr. Rao asked for a refund of $1,249.99 on March 3rd, 2025; call +91 98765 43210.",
    "Book 2 seats for Priyonka on 2026-12-01 at 9 am, 12 km away, order No. 4471.",
]


def frame(ms: float, rate: int) -> bytes:
    count = int(rate * ms / 1000)
    return array("h", (int(8_000 * math.sin(i / 7)) for i in range(count))).tobytes()


def median_ms(fn: Callable[[Any], Any], inputs: list[Any], rounds: int = 300) -> float:
    times = []
    for i in range(rounds):
        value = inputs[i % len(inputs)]
        start = time.perf_counter_ns()
        fn(value)
        times.append(time.perf_counter_ns() - start)
    return statistics.median(times) / 1e6


def names() -> KnownNames:
    known = ["Rahul", "Priyanka", "Acme Corp", "Hitesh", "Rao"]
    return KnownNames(
        known + [f"Person{i} Surname{i}" for i in range(100)] + [f"Name{i}" for i in range(100)]
    )


def cold_names(text: str) -> str:
    fixer = NAMES
    fixer._memo.clear()  # no cached words: every word is matched afresh
    return fixer.fix(text)


NAMES = names()
UNSURE = UnsureWords()
REQUEST = ApprovalRequest(
    approval_id="a", run_id="r", call_id="c", tool="pay", arguments={"amount": 450000}
)


def hear_and_read_back(text: str) -> str | None:
    UNSURE.hear([Word(w, 0.5) for w in text.split()], run_id="r")
    return UNSURE.readback(REQUEST)


CHECKS = {
    "input frame (20 ms, 16 kHz)": (InputCheck(gate_db=-50), [frame(20, 16_000)]),
    "output frame (20 ms, 48 kHz)": (OutputCheck(rate=48_000).feed, [frame(20, 48_000)]),
    "spoken form (en-IN)": (SpokenForm(locale="en-IN"), SENTENCES),
    "spoken form (en-US)": (SpokenForm(pronounce={"VoxCat": "vox cat"}), SENTENCES),
    "known names, 205 names, cold": (cold_names, SENTENCES),
    "unsure words and read-back": (hear_and_read_back, SENTENCES),
}


@pytest.mark.parametrize("name", CHECKS)
def test_each_check_stays_under_a_millisecond(name: str) -> None:
    fn, inputs = CHECKS[name]
    median_ms(fn, inputs, rounds=20)  # warm up
    took = median_ms(fn, inputs)
    assert took < BUDGET_MS, f"{name}: median {took:.3f} ms"
