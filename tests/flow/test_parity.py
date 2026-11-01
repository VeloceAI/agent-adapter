import pytest

from tests.flow.parity import (
    CASES,
    assert_final_matches,
    assert_pauses_match,
    assert_state_matches,
    case,
    run_local,
)
from tests.flow.parity_flows import review_flow


@pytest.mark.parametrize("parity", CASES, ids=[c.name for c in CASES])
async def test_local_runner_matches_each_parity_case(parity):
    if parity.raises is not None:
        with pytest.raises(parity.raises):
            await run_local(parity)
        return
    pauses, final = await run_local(parity)
    assert_pauses_match(parity, pauses)
    assert_final_matches(parity, final)


def test_case_lookup_and_unique_names():
    assert [c.name for c in CASES] == list(dict.fromkeys(c.name for c in CASES))
    assert case("loop_exhausted").visits["work"] == 4


def test_append_fields_compare_as_multisets_and_keyed_fields_by_key():
    want = case("fanout_keyed").expected
    reordered = {
        **want,
        "log": list(reversed(want["log"])),
        "evidence": list(reversed(want["evidence"])),
    }
    assert_state_matches(case("fanout_keyed").flow, reordered, want)
    for broken in (
        {**want, "log": [*want["log"], "extra"]},
        {**want, "evidence": want["evidence"][:2]},
        {**want, "summary": "other"},
    ):
        with pytest.raises(AssertionError):
            assert_state_matches(case("fanout_keyed").flow, broken, want)
    with pytest.raises(AssertionError, match="fields differ"):
        assert_state_matches(review_flow, {"topic": "x"}, case("pause_denied").expected)


def test_keyed_collision_compares_the_winning_value():
    parity = case("keyed_collision")
    loser = {**parity.expected, "hits": [{"source_id": "x", "text": "alpha"}]}
    with pytest.raises(AssertionError, match="hits"):
        assert_state_matches(parity.flow, loser, parity.expected)
