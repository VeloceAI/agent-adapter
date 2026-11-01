"""schema._names.unique and the field names of loose models: numbered clashes in O(1) each."""

from agent_adapter.schema._names import unique
from agent_adapter.schema.models import _field_names
from tests.work import run_child


def test_unique_keeps_a_free_name_and_numbers_clashes_from_2():
    taken, counters = {"a"}, {}
    assert unique("b", taken, counters) == "b"
    names = []
    for _ in range(3):
        names.append(unique("a", taken, counters))
        taken.add(names[-1])
    assert names == ["a_2", "a_3", "a_4"]


def test_unique_skips_variants_already_taken_and_reoffers_an_untaken_one():
    taken, counters = {"a", "a_2", "a_3"}, {}
    assert unique("a", taken, counters) == "a_4"
    assert unique("a", taken, counters) == "a_4"  # not taken by the caller: offered again


def test_unique_cuts_variants_to_max_len_and_shares_the_stem_counter():
    long = "x" * 64
    taken, counters = {long}, {}
    first = unique(long, taken, counters, max_len=64)
    assert first == "x" * 62 + "_2" and len(first) == 64
    taken.add(first)
    other = "x" * 63 + "y"  # same 56-character stem
    taken.add(other)
    assert unique(other, taken, counters, max_len=64) == "x" * 62 + "_3"


def test_field_names_are_unchanged_for_ordinary_schemas():
    props = ["first-name", "first_name", "a b", "a.b", "1st", "", "_x", "model_y", "ok"]
    assert _field_names(props) == {
        "first_name": "first_name",
        "ok": "ok",
        "first-name": "first_name_2",
        "a b": "a_b",
        "a.b": "a_b_2",
        "1st": "p_1st",
        "": "p_",
        "_x": "p__x",
        "model_y": "p_model_y",
    }


# Lines of agent_adapter run per name (deterministic, unlike seconds): ~29 with per-stem
# counters; linear probing runs ~n / 2 per name (about 9 M lines for 3000 names).
LINES_PER_NAME = 100
CHILD = f"""
import itertools, json, string
from tests.work import counting
from agent_adapter.schema.models import _field_names
marks = string.punctuation.replace("_", "")
clashing = ["x" + "".join(p) for p in itertools.product(marks, repeat=3)]  # all "x___"
out = {{}}
for n in (3000, 10000):
    with counting(n * {LINES_PER_NAME}):
        names = _field_names(clashing[:n])
    out[n] = list(names.values())
print(json.dumps(out))
"""


def test_thousands_of_clashing_property_names_are_named_in_linear_work():
    result = run_child(CHILD)
    for n in ("3000", "10000"):
        names = result[n]
        assert names[:3] == ["x___", "x____2", "x____3"] and names[-1] == f"x____{n}"
        assert len(set(names)) == int(n)
