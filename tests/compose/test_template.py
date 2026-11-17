import pytest

from agent_adapter.compose import Blank, Template, check_value
from agent_adapter.compose.blanks import render
from agent_adapter.compose.checks import from_markdown
from agent_adapter.compose.template import slots
from agent_adapter.errors import FlowDefinitionError


def test_slots_find_names_skip_escapes_and_report_bad_ones():
    names, bad = slots("# {{ title }} and {{summary}} \\{{ literal }} {{ two words }}")
    assert names == ["title", "summary"]
    assert bad == ["{{ two words }}"]


def test_order_comes_from_blanks_used_in_prompts():
    template = Template(
        "{{ intro }}\n{{ facts }}\n{{ verdict }}",
        [
            Blank("intro", "Introduce {{ topic }}"),
            Blank("facts", "List facts about {{ topic }}", "list"),
            Blank("verdict", "Judge, given {{ intro }} and {{ facts }}"),
        ],
    )
    assert template.layers() == [["intro", "facts"], ["verdict"]]
    assert template.inputs == ("topic",)
    assert template.users("facts") == ["verdict"]
    assert template.missing_inputs({}) == ["topic"]


def test_a_helper_blank_used_only_in_prompts_is_allowed():
    template = Template("{{ body }}", [Blank("notes", "Notes"), Blank("body", "Use {{ notes }}")])
    assert template.layers() == [["notes"], ["body"]]


def test_every_problem_is_reported_together():
    with pytest.raises(FlowDefinitionError) as caught:
        Template(
            "{{ a }} {{ bad name }}",
            [
                Blank("a", "x"),
                Blank("a", "y"),
                Blank("t", "rows", "table"),
                Blank("aa_editor", "z"),
                Blank("1st", ""),
            ],
        )
    text = "\n".join(caught.value.problems)
    for expected in (
        "defined twice",
        "needs columns",
        "reserved",
        "must be letters",
        "prompt is empty",
        "is not a blank name",
        "is not used",
    ):
        assert expected in text


def test_a_cycle_is_refused():
    with pytest.raises(FlowDefinitionError, match="wait for each other"):
        Template("{{ a }} {{ b }}", [Blank("a", "{{ b }}"), Blank("b", "{{ a }}")])


def test_render_fills_blanks_and_inputs_and_unescapes():
    template = Template("# {{ title }}\n{{ body }} \\{{ kept }}", [Blank("body", "Write")])
    assert template.render({"body": "Hi."}, {"title": "Q3"}) == "# Q3\nHi. {{ kept }}"


def test_tables_and_diagrams_are_drawn_by_the_code():
    table = Blank("t", "rows", "table", columns=["Name", "Note"])
    drawn = render(table, [{"Name": "a|b", "Note": "two\nlines"}])
    assert drawn.splitlines() == ["| Name | Note |", "| --- | --- |", "| a\\|b | two lines |"]
    diagram = Blank("d", "flow", "diagram", direction="LR")
    value = {
        "nodes": [{"id": "x", "label": 'say "hi" <b>'}, {"id": "y", "label": ""}],
        "edges": [{"source": "x", "target": "y", "label": "next"}],
    }
    assert render(diagram, value).splitlines() == [
        "```mermaid",
        "flowchart LR",
        '  n1["say #quot;hi#quot; #lt;b#gt;"]',
        '  n2["y"]',
        '  n1 -->|"next"| n2',
        "```",
    ]
    assert render(Blank("l", "x", "list"), ["a", "b"]) == "- a\n- b"
    assert render(Blank("n", "x", "number"), 3.0) == "3"


def test_checks():
    text = Blank("s", "x", max_words=3)
    assert check_value(text, "one two three four") == ["the answer has 4 words; the limit is 3"]
    assert check_value(text, "  ") == ["the answer is empty"]
    assert check_value(Blank("s", "x", required=False), "") == []
    table = Blank("t", "x", "table", columns=["A", "B"])
    assert check_value(table, [{"A": "1", "C": "2"}]) == [
        "row 1 lacks columns ['B']",
        "row 1 has unknown columns ['C']",
    ]
    diagram = Blank("d", "x", "diagram")
    bad = {
        "nodes": [{"id": "a", "label": ""}],
        "edges": [{"source": "a", "target": "z", "label": ""}],
    }
    assert check_value(diagram, bad) == ["edge 1 links unknown nodes ['z']"]


@pytest.mark.parametrize(
    ("blank", "value"),
    [
        (Blank("t", "x", "table", columns=["A", "B"]), [{"A": "1 | 2", "B": "x"}]),
        (Blank("l", "x", "list"), ["one", "two"]),
        (Blank("n", "x", "number"), 2.5),
        (
            Blank("d", "x", "diagram"),
            {
                "nodes": [{"id": "n1", "label": 'a "b"'}, {"id": "n2", "label": "c"}],
                "edges": [{"source": "n1", "target": "n2", "label": "go"}],
            },
        ),
    ],
)
def test_markdown_reads_back_to_the_same_value(blank, value):
    assert from_markdown(blank, render(blank, value)) == (value, [])


def test_broken_markdown_is_not_read_back():
    table = Blank("t", "x", "table", columns=["A", "B"])
    assert from_markdown(table, "| A | C |\n| --- | --- |")[1] == [
        "the revision changed the columns to ['A', 'C']"
    ]
    assert from_markdown(Blank("d", "x", "diagram"), "graph TD\nA-->B")[1]
    assert from_markdown(Blank("l", "x", "list"), "just text")[1]
    assert from_markdown(Blank("n", "x", "number"), "many")[1]
