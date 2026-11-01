from typing import Annotated

import pytest
from pydantic import BaseModel

from agent_adapter.core.errors import FlowDefinitionError, FlowError
from agent_adapter.flow.state import Merge, MergeRule, apply_updates, keyed, merge_rules


class Hit(BaseModel):
    source_id: str
    text: str = ""


class State(BaseModel):
    title: str = ""
    log: Annotated[list[str], Merge.append] = []
    total: Annotated[int, Merge.sum] = 0
    hits: Annotated[list[Hit], keyed(by="source_id")] = []
    rows: Annotated[list[dict], Merge.keyed("id")] = []


RULES = merge_rules(State)


def test_merge_rules_reads_annotations_and_defaults_to_replace():
    assert {name: rule.name for name, rule in RULES.items()} == {
        "title": "replace",
        "log": "append",
        "total": "sum",
        "hits": "keyed(source_id)",
        "rows": "keyed(id)",
    }
    assert all(isinstance(rule, MergeRule) for rule in RULES.values())


def test_merge_rules_rejects_two_rules_on_one_field():
    class Twice(BaseModel):
        log: Annotated[list[str], Merge.append, Merge.sum] = []

    with pytest.raises(FlowDefinitionError) as info:
        merge_rules(Twice)
    assert info.value.problems == ["field 'log' has more than one merge rule"]


def test_replace_keeps_equal_values_and_rejects_conflicts():
    assert Merge.replace.merge("old", ["a", "a"]) == "a"
    with pytest.raises(FlowError, match="conflicting writes"):
        Merge.replace.merge("old", ["a", "b"])


def test_replace_conflict_names_the_field():
    with pytest.raises(FlowError, match="conflicting writes to title"):
        apply_updates(State(), [{"title": "a"}, {"title": "b"}], RULES)


def test_append_extends_lists_and_adds_single_items_in_update_order():
    assert Merge.append.merge(["x"], [["a", "b"], "c", []]) == ["x", "a", "b", "c"]
    assert Merge.append.merge(None, ["a"]) == ["a"]


def test_sum_adds_onto_the_current_value():
    assert Merge.sum.merge(2, [3, 4]) == 9
    assert Merge.sum.merge(None, [1.5]) == 1.5


def test_keyed_later_update_replaces_earlier_and_keeps_first_seen_order():
    rule = Merge.keyed("id")
    current = [{"id": 1, "v": "old"}, {"id": 2, "v": "two"}]
    merged = rule.merge(
        current, [[{"id": 3, "v": "three"}, {"id": 1, "v": "new"}], {"id": 3, "v": "3"}]
    )
    assert merged == [{"id": 1, "v": "new"}, {"id": 2, "v": "two"}, {"id": 3, "v": "3"}]


def test_keyed_reads_model_attributes():
    merged = keyed("source_id").merge([], [[Hit(source_id="a"), Hit(source_id="a", text="x")]])
    assert merged == [Hit(source_id="a", text="x")]


def test_keyed_rejects_an_item_without_the_key():
    with pytest.raises(FlowError, match="item without key 'id'"):
        Merge.keyed("id").merge([], [{"name": "x"}])


def test_keyed_needs_a_key_name():
    with pytest.raises(FlowDefinitionError):
        Merge.keyed("")


def test_apply_updates_merges_in_order_and_revalidates():
    state = State(log=["start"], total=1)
    out = apply_updates(
        state,
        [
            {"log": ["a"], "total": 2, "hits": [{"source_id": "s1", "text": "one"}]},
            {"log": "b", "total": 3, "hits": {"source_id": "s1", "text": "uno"}},
            {"title": "done"},
        ],
        RULES,
    )
    assert out.log == ["start", "a", "b"]
    assert out.total == 6
    assert out.hits == [Hit(source_id="s1", text="uno")]
    assert out.title == "done"
    assert state.log == ["start"]  # the input state is not changed


def test_apply_updates_without_updates_returns_an_equal_state():
    state = State(title="t")
    assert apply_updates(state, [], RULES) == state


def test_apply_updates_rejects_unknown_keys():
    with pytest.raises(FlowError, match=r"unknown state fields \['nope', 'zzz'\]"):
        apply_updates(State(), [{"zzz": 1}, {"nope": 2, "title": "x"}], RULES)


def test_apply_updates_reports_validation_errors_without_values():
    with pytest.raises(FlowError, match="state update is invalid") as info:
        apply_updates(State(), [{"total": "secret-value"}], {})
    assert "secret-value" not in str(info.value)


def test_apply_updates_wraps_type_errors_from_a_rule():
    with pytest.raises(FlowError, match="cannot merge total with rule sum"):
        apply_updates(State(), [{"total": "x"}], RULES)


def test_apply_updates_uses_replace_for_fields_without_a_rule():
    assert apply_updates(State(), [{"log": ["a"]}, {"log": ["a"]}], {}).log == ["a"]
