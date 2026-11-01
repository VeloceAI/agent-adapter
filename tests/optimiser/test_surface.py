import pytest

from agent_adapter.core.errors import OptimiserError
from agent_adapter.optimiser.model import Edit, HarnessTree, SurfaceCaps
from agent_adapter.optimiser.surface import (
    DEFAULT_CAPS,
    always_on_chars,
    cap_key,
    dimension_problems,
    edits_between,
    parse_path,
    validate_tree,
)

ALL_KINDS = frozenset({"tool", "hook", "prompt", "flow"})


@pytest.mark.parametrize(
    ("path", "kind", "name", "field", "node"),
    [
        ("tool/search/description", "tool", "search", "description", ""),
        ("tool/search_v2/prompt_hint", "tool", "search_v2", "prompt_hint", ""),
        ("hook/budget/config.json", "hook", "budget", "config.json", ""),
        ("hook/My.Hook-1/config.json", "hook", "My.Hook-1", "config.json", ""),
        ("prompt/system", "prompt", "system", "", ""),
        ("prompt/layout/few_shots", "prompt", "layout", "few_shots", ""),
        ("flow/research/plan/prompt", "flow", "research", "prompt", "plan"),
    ],
)
def test_parse_path_accepts_each_kind(path, kind, name, field, node):
    parsed = parse_path(path)
    assert (parsed.kind, parsed.name, parsed.field, parsed.node) == (kind, name, field, node)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/prompt/system",
        "prompt/system/",
        "Prompt/system",
        "tool/../description",
        "tool/a/../../etc",
        "tool/a\\b/description",
        "tool/a b/description",
        "tool/a/summary",
        "tool/1abc/description",
        "hook/../config.json",
        "hook/x/config.yaml",
        "prompt/layout/",
        "prompt/other",
        "flow/f/n/notes",
        "flow/f/n-1/prompt",
        "scripts/run.py",
    ],
)
def test_parse_path_rejects_bad_paths(path):
    with pytest.raises(OptimiserError, match="not an editable path"):
        parse_path(path)


def test_cap_key_maps_paths_to_patterns():
    assert cap_key("tool/x/description") == "tool/*/description"
    assert cap_key("tool/x/prompt_hint") == "tool/*/prompt_hint"
    assert cap_key("hook/x/config.json") == "hook/*/config.json"
    assert cap_key("prompt/system") == "prompt/system"
    assert cap_key("prompt/layout/s") == "prompt/layout/*"
    assert cap_key("flow/f/n/prompt") == "flow/*/*/prompt"
    assert set(DEFAULT_CAPS) == {
        "tool/*/description",
        "tool/*/prompt_hint",
        "hook/*/config.json",
        "prompt/system",
        "prompt/layout/*",
        "flow/*/*/prompt",
    }


def test_dimension_problems_reject_off_dimension_edits():
    edits = [
        Edit(path="tool/a/description", content="x", reason="r"),
        Edit(path="prompt/system", content="y", reason="r"),
        Edit(path="bad path", content="z", reason="r"),
    ]
    assert dimension_problems(edits, "procedure") == [
        "prompt/system: the procedure editor may only edit tool/*, hook/* paths",
        "bad path: not an editable path",
    ]
    assert dimension_problems(edits[1:2], "knowledge") == []
    assert dimension_problems(edits[:1], "knowledge") == [
        "tool/a/description: the knowledge editor may only edit prompt/*, flow/* paths"
    ]


def caps(**extra) -> SurfaceCaps:
    return SurfaceCaps(max_chars=dict(DEFAULT_CAPS), kinds=ALL_KINDS, **extra)


def test_validate_tree_lists_every_problem_together():
    tree = HarnessTree(
        files={
            "prompt/system": "x" * (DEFAULT_CAPS["prompt/system"] + 1),
            "tool/a/description": "fine",
            "nope": "z",
            "hook/budget/config.json": "[1, 2]",
            "hook/other/config.json": "{bad",
        }
    )
    problems = validate_tree(tree, caps())
    assert problems == [
        "hook/budget/config.json: config must be a JSON object",
        "hook/other/config.json: config is not valid JSON (Expecting property name enclosed in "
        "double quotes: line 1 column 2 (char 1))",
        "nope: not an editable path",
        f"prompt/system: {DEFAULT_CAPS['prompt/system'] + 1} chars, cap is "
        f"{DEFAULT_CAPS['prompt/system']}",
    ]


def test_validate_tree_rejects_kinds_the_host_does_not_offer():
    tree = HarnessTree(files={"flow/f/n/prompt": "p"})
    limited = SurfaceCaps(max_chars={}, kinds=frozenset({"tool"}))
    assert validate_tree(tree, limited) == ["flow/f/n/prompt: kind 'flow' is not editable here"]


def test_validate_tree_checks_hook_config_against_its_schema():
    schema = {
        "type": "object",
        "properties": {"max_usd": {"type": "number", "minimum": 0}},
        "required": ["max_usd"],
        "additionalProperties": False,
    }
    tree = HarnessTree(files={"hook/budget/config.json": '{"max_usd": -1, "x": 1}'})
    assert validate_tree(tree, caps(hook_schemas={"budget": schema})) == [
        "hook/budget/config.json: $: unexpected key 'x'",
        "hook/budget/config.json: $.max_usd: -1 is below the minimum 0",
    ]


class _Host:
    def __init__(self, result):
        self.result = result

    def validate(self, tree):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_validate_tree_adds_host_problems_and_survives_a_broken_host():
    tree = HarnessTree(files={"prompt/system": "s"})
    assert validate_tree(tree, caps(), _Host(["system prompt must mention tax"])) == [
        "system prompt must mention tax"
    ]
    assert validate_tree(tree, caps(), _Host(RuntimeError("boom"))) == [
        "host validation failed: RuntimeError"
    ]


def test_edits_between_round_trips():
    a = HarnessTree(files={"p/1": "a", "p/2": "b", "p/3": "c"})
    b = HarnessTree(files={"p/1": "a", "p/2": "B", "p/4": "d"})
    edits = edits_between(a, b)
    assert edits == {"p/2": "B", "p/3": None, "p/4": "d"}
    assert a.with_edits(edits) == b


def test_always_on_chars_counts_prompts_and_hints_only():
    tree = HarnessTree(
        files={
            "prompt/system": "12345",
            "prompt/layout/s": "123",
            "tool/a/prompt_hint": "12",
            "tool/a/description": "not always on",
            "flow/f/n/prompt": "node only",
        }
    )
    assert always_on_chars(tree) == 10
