import asyncio
import json
import logging

import pytest

from agent_adapter.core.errors import ConfigError, ModelUnavailable
from agent_adapter.models.callable import CallableModel
from agent_adapter.optimiser import CaseRun, EvalCase, Score
from agent_adapter.optimiser.graders import (
    all_of,
    code_check,
    contains,
    exact,
    llm_judge,
    regex,
)


def case(expected="80C"):
    return EvalCase(id="q", input="tuition fees", expected=expected)


def run(output):
    return CaseRun(case_id="q", label="b", ok=True, delivered=True, output=output, transcript="")


async def grade(judge, output, expected="80C"):
    return await judge(case(expected), run(output))


async def test_exact_normalises_case_space_and_trailing_punctuation():
    assert (await grade(exact(), "  80c. ")).value == 1.0
    miss = await grade(exact(), "Section 80C")
    assert (miss.value, miss.passed, miss.notes) == (0.0, False, "not the expected answer")
    assert (await grade(exact(normalise=False), "80c")).value == 0.0
    assert (await grade(exact(), "80D", ["80C", "80D"])).passed is True


async def test_contains_finds_the_expected_text():
    assert (await grade(contains(), "That is Section 80C.")).passed is True
    assert (await grade(contains(), "a deduction may apply")).passed is False
    assert (await grade(contains(normalise=False), "section 80c")).passed is False


async def test_regex_uses_expected_as_the_pattern():
    assert (await grade(regex(), "SECTION 80C", r"section\s+80c")).passed is True
    assert (await grade(regex(flags=0), "SECTION 80C", r"section")).passed is False


async def test_no_expected_value_or_a_bad_pattern_is_ungraded(caplog):
    caplog.set_level(logging.WARNING)
    assert await grade(contains(), "80C", None) is None
    assert "has no expected value" in caplog.text
    assert await grade(regex(), "x", "(") is None
    assert "regex grader failed" in caplog.text


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, 1.0), (False, 0.0), (0.5, 0.5), (1, 1.0), (Score(value=0.2, notes="n"), 0.2)],
)
async def test_code_check_accepts_bool_float_and_score(value, expected):
    assert (await grade(code_check(lambda c, r: value), "x")).value == expected


async def test_code_check_awaits_and_rejects_out_of_range_or_raising_checks(caplog):
    async def later(c, r):
        return True

    assert (await grade(code_check(later), "x")).passed is True
    assert await grade(code_check(lambda c, r: 1.5), "x") is None
    assert await grade(code_check(lambda c, r: 1 / 0), "x") is None
    assert "code_check grader failed" in caplog.text


def judge_model(reply, seen=None):
    async def answer(request):
        if seen is not None:
            seen.append(request)
        if isinstance(reply, BaseException):
            raise reply
        return json.dumps(reply)

    return CallableModel(answer)


async def test_llm_judge_parses_the_schema_and_scrubs_expected_from_notes():
    seen = []
    model = judge_model({"value": 0.9, "passed": True, "notes": "names 80c correctly"}, seen)
    score = await grade(llm_judge(model, "Names the right section."), "Section 80C")
    assert score == Score(
        value=0.9, passed=True, notes="names [expected answer withheld] correctly"
    )
    prompt = seen[0].messages[1].content
    assert "<expected>\n80C\n</expected>" in prompt and seen[0].stage == "judge"
    assert seen[0].output_schema.name == "judge_score"


async def test_llm_judge_can_hide_expected_and_list_several():
    seen = []
    model = judge_model({"value": 1, "passed": True, "notes": "ok"}, seen)
    await grade(llm_judge(model, "r", show_expected=False, stage="grade"), "x")
    assert "<expected>" not in seen[0].messages[1].content and seen[0].stage == "grade"
    await grade(llm_judge(model, "r"), "x", ["80C", "80D"])
    assert "- 80C\n- 80D" in seen[1].messages[1].content


async def test_llm_judge_model_errors_and_bad_replies_are_ungraded(caplog):
    for reply in (
        ModelUnavailable("down", provider="judge"),
        {"value": 3, "passed": True, "notes": ""},
    ):
        assert await grade(llm_judge(judge_model(reply), "r"), "x") is None
    assert caplog.text.count("llm_judge grader failed") == 2


def test_llm_judge_needs_a_rubric():
    with pytest.raises(ConfigError):
        llm_judge(judge_model({}), " ")


async def test_all_of_weights_the_graded_ones_and_ignores_none():
    async def ungraded(c, r):
        return None

    async def odd(c, r):
        return "nope"

    judge = all_of(contains(), code_check(lambda c, r: False), ungraded, odd, weights=[3, 1, 1, 1])
    score = await grade(judge, "80C")
    assert (score.value, score.passed) == (0.75, False)
    assert score.notes == "the check failed; all_of[2] not graded; all_of[3] not graded"
    assert await grade(all_of(ungraded), "x") is None
    assert (await grade(all_of(code_check(lambda c, r: 0.4)), "x")).passed is None
    assert (await grade(all_of(contains(), odd), "80C")).passed is False  # fail closed


async def test_all_of_survives_a_raising_judge_but_not_cancellation():
    async def broken(c, r):
        raise RuntimeError("boom")

    async def cancelled(c, r):
        raise asyncio.CancelledError

    assert (await grade(all_of(broken, contains()), "80C")).value == 1.0
    assert (await grade(all_of(broken, contains()), "80C")).passed is False
    assert (await grade(all_of(broken, contains()), "no")).passed is False
    raising_check = code_check(lambda c, r: 1 / 0)
    assert (await grade(all_of(raising_check, contains()), "80C")).passed is False
    with pytest.raises(asyncio.CancelledError):
        await grade(all_of(cancelled), "x")


@pytest.mark.parametrize("weights", [[1.0], [-1.0, 1.0], [0.0, 0.0], [float("nan"), 1.0]])
def test_all_of_rejects_bad_weights(weights):
    with pytest.raises(ConfigError):
        all_of(contains(), exact(), weights=weights)
    with pytest.raises(ConfigError):
        all_of()


@pytest.mark.parametrize("make", [exact, contains, regex])
async def test_every_text_grader_leaves_a_case_without_expected_ungraded(make):
    assert await grade(make(), "80C", None) is None


async def test_llm_judge_shows_a_structured_input_as_json():
    seen = []
    model = judge_model({"value": 1, "passed": True, "notes": ""}, seen)
    await llm_judge(model, "r")(EvalCase(id="q", input={"topic": "rent"}), run("x"))
    assert '<input>\n{"topic": "rent"}\n</input>' in seen[0].messages[1].content
