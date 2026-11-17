from agent_adapter.compose import Blank, Editor, Template, compose
from agent_adapter.tracing import EventLog
from tests.compose.helpers import Script, executor, parsed

ROWS = {"rows": [{"Region": "EU", "Sales": "10"}, {"Region": "US", "Sales": "12"}]}
NODES = [{"id": "a", "label": "Lead"}, {"id": "b", "label": "Deal"}]
GRAPH = {"nodes": NODES, "edges": [{"source": "a", "target": "b", "label": ""}]}
TABLE_MD = "| Region | Sales |\n| --- | --- |\n| EU | 10 |\n| US | 12 |"


def document() -> Template:
    return Template(
        "{{ intro }}\n\n{{ sales }}\n\n{{ funnel }}",
        [
            Blank("intro", "Intro", max_words=10),
            Blank("sales", "Sales", "table", columns=["Region", "Sales"]),
            Blank("funnel", "Funnel", "diagram"),
        ],
        name="doc",
    )


def revisions(*items: tuple[str, str]) -> dict:
    found = [{"blank": blank, "markdown": text, "reason": "fit"} for blank, text in items]
    return parsed({"revisions": found})


async def run(edits: dict, **options):
    answers = {"intro": "we sold a lot", "sales": parsed(ROWS), "funnel": parsed(GRAPH)}
    script = Script({**answers, "editor": edits})
    editor = Editor(script.model, style="formal")
    result = await compose(
        document(), model=script.model, editor=editor, executor=executor(**options)
    )
    return result, script


async def test_the_editor_rewords_text_and_records_before_and_after():
    result, script = await run(revisions(("intro", "Sales were strong.")))
    assert result.markdown.startswith("Sales were strong.\n\n| Region | Sales |")
    (revision,) = result.revisions
    assert (revision.before, revision.after, revision.accepted) == (
        "we sold a lot",
        "Sales were strong.",
        True,
    )
    assert result.blanks["intro"].edited and not result.blanks["sales"].edited
    sent = script.calls("editor")[0]
    assert "Style: formal." in sent.messages[0].content
    assert '<part name="sales">' in sent.messages[1].content
    assert script.order()[-1] == "editor"


async def test_the_editor_cannot_break_a_table_or_a_diagram():
    dropped_row = "| Region | Sales |\n| --- | --- |\n| EU | 10 |"
    renamed = TABLE_MD.replace("Sales |\n", "Revenue |\n", 1)
    extra_edge = '```mermaid\nflowchart TD\n  n1["Lead"]\n  n2["Deal"]\n  n2 --> n1\n```'
    result, _ = await run(
        revisions(
            ("sales", dropped_row),
            ("sales", renamed),
            ("funnel", extra_edge),
            ("funnel", "a picture of a funnel"),
            ("ghost", "boo"),
            ("intro", " ".join(["long"] * 20)),
        )
    )
    assert [r.accepted for r in result.revisions] == [False] * 6
    problems = [r.problems[0] for r in result.revisions]
    assert problems == [
        "the revision has 1 rows, not 2",
        "the revision changed the columns to ['Region', 'Revenue']",
        "the revision changed the diagram's nodes or edges",
        "the revision is not a mermaid block",
        "there is no such part",
        "the answer has 20 words; the limit is 10",
    ]
    assert TABLE_MD in result.markdown and "  n1 --> n2" in result.markdown
    assert not any(b.edited for b in result.blanks.values())


async def test_the_editor_may_reword_cells_and_labels():
    cells = TABLE_MD.replace("US", "United States")
    labels = '```mermaid\nflowchart TD\n  n1["New lead"]\n  n2["Deal"]\n  n1 --> n2\n```'
    result, _ = await run(revisions(("sales", cells), ("funnel", labels)))
    assert [r.accepted for r in result.revisions] == [True, True]
    assert result.blanks["sales"].value[1] == {"Region": "United States", "Sales": "12"}
    assert '  n1["New lead"]' in result.markdown


async def test_an_unreadable_editor_answer_changes_nothing():
    events = EventLog()
    result, _ = await run("not json at all", sink=events)
    assert result.revisions == () and result.markdown.startswith("we sold a lot")
    (edited,) = [e for e in events.events if e.type == "compose.edited"]
    assert (edited.data["revised"], edited.data["rejected"]) == (0, 0)
