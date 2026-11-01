import json

import pytest

from agent_adapter.cache.layout import PromptLayout
from agent_adapter.core.errors import ConfigError
from agent_adapter.optimiser import RegistryHost, apply_tree, layout_from_tree, load_harness
from agent_adapter.optimiser.model import HarnessTree
from tests.optimiser.fakes import CASES, make_registry
from tests.optimiser.loop_support import CITE, run


async def test_load_harness_round_trips_harness_after_json(tmp_path):
    from tests.optimiser.fakes import proposal

    result, *_ = await run(tmp_path, {"know1": proposal(CITE)})
    assert result.chosen == "agg1"
    tree = load_harness(tmp_path / "out")
    assert tree == result.candidates["agg1"].tree
    assert tree.files["prompt/system"] == CITE[1]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{not json", "cannot read"),
        (json.dumps({"version": 2, "files": {}}), "version 1"),
        (json.dumps({"version": 1, "files": {"prompt/system": 3}}), "map paths to text"),
        (json.dumps({"version": 1, "files": {"../x": "y"}}), "not an editable path"),
    ],
)
def test_load_harness_rejects_bad_files(tmp_path, text, message):
    (tmp_path / "harness.after.json").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_harness(tmp_path)


def test_load_harness_without_a_run(tmp_path):
    with pytest.raises(ConfigError, match="run the optimiser first"):
        load_harness(tmp_path)


def test_layout_from_tree_equals_the_registry_hosts_layout():
    registry = make_registry()
    base = PromptLayout(static="Answer.", dynamic="today", sections={"style": "Be brief."})
    host = RegistryHost(registry, layout=base, cases=CASES)
    for tree in (
        host.surface(),
        host.surface().with_edits(
            {"prompt/system": "Cite.", "tool/fetch/prompt_hint": "Only after search."}
        ),
    ):
        expected = host.layout_for(tree, apply_tree(registry, tree))
        assert layout_from_tree(base, tree) == expected
    assert layout_from_tree(base, tree).dynamic == "today"
    assert "- fetch: Only after search." in layout_from_tree(base, tree).sections["aa.tool_hints"]


def test_layout_from_tree_without_hints_has_no_hint_section():
    tree = HarnessTree(files={"prompt/system": "S", "prompt/layout/a": "A"})
    layout = layout_from_tree(PromptLayout(static="old"), tree)
    assert (layout.static, layout.sections) == ("S", {"a": "A"})
