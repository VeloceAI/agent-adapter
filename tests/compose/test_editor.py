import json

from agent_adapter.compose import Blank, Editor, Template, compose
from agent_adapter.tracing import EventLog
from tests.compose.helpers import Script, executor, parsed, prompt_of

ROWS = {"rows": [{"Region": "EU", "Sales": "10"}, {"Region": "US", "Sales": "12"}]}
TABLE_MD = "| Region | Sales |\n| --- | --- |\n| EU | 10 |\n| US | 12 |"
SOURCES = [{"id": "q3", "text": "Sales rose in Europe."}]


def document() -> Template:
    return Template(
        "{{ intro }}\n\n{{ sales }}\n\n{{ outlook }}",
        [
            Blank("intro", "Intro", max_words=10),
            Blank("sales", "Sales", "table", columns=["Region", "Sales"]),
            Blank("outlook", "Outlook"),
        ],
        name="doc",
    )


def revisions(*items: tuple[str, str]) -> dict:
    found = [{"blank": blank, "text": text, "reason": "fit"} for blank, text in items]
    return parsed({"revisions": found})


async def run(edits, *, min_words: int = 0, **options):
    answers = {
        "intro": "we sold a [lot](cite:q3)",
        "sales": parsed(ROWS),
        "outlook": "Next quarter looks fine.",
    }
    script = Script({**answers, "aa_editor": edits})
    editor = Editor(script.model, style="formal", min_words=min_words)
    result = await compose(
        document(),
        model=script.model,
        editor=editor,
        verifier=False,
        sources=SOURCES,
        executor=executor(**options),
    )
    return result, script


async def test_the_editor_sees_only_text_blanks_and_returns_only_changes():
    result, script = await run(revisions(("intro", "We sold [a lot](cite:q3).")))
    assert result.markdown.startswith("We sold a lot [1].\n\n| Region | Sales |")
    (revision,) = result.revisions
    assert (revision.before, revision.after, revision.accepted) == (
        "we sold a [lot](cite:q3)",
        "We sold [a lot](cite:q3).",
        True,
    )
    assert result.blanks["intro"].edited and not result.blanks["outlook"].edited
    sent = script.calls("aa_editor")[0]
    assert "Style: formal." in sent.messages[0].content
    assert json.loads(prompt_of(sent)) == {
        "intro": "we sold a [lot](cite:q3)",
        "outlook": "Next quarter looks fine.",
    }
    assert script.order()[-1] == "aa_editor"


async def test_a_revision_that_changes_citations_or_fails_checks_is_rejected():
    result, _ = await run(
        revisions(
            ("intro", "We sold a lot."),
            ("intro", "We sold [a lot](cite:q3) [here](cite:q4)."),
            ("sales", "| Region |"),
            ("intro", " ".join(["long"] * 20) + " [x](cite:q3)"),
        )
    )
    assert [r.accepted for r in result.revisions] == [False] * 4
    assert [r.problems[0] for r in result.revisions] == [
        "the revision drops citations ['q3']",
        "the revision adds citations ['q4']",
        "there is no such text part",
        "the answer has 21 words; the limit is 10",
    ]
    assert TABLE_MD in result.markdown and "we sold a lot [1]" in result.markdown
    assert not any(b.edited for b in result.blanks.values())


async def test_the_editor_is_skipped_for_short_documents():
    result, script = await run(revisions(("intro", "x")), min_words=150)
    assert script.calls("aa_editor") == [] and result.revisions == ()
    single = Template("{{ a }}", [Blank("a", "A")])
    lone = Script({"a": " ".join(["word"] * 200)})
    await compose(single, model=lone.model, executor=executor())
    assert lone.order() == ["a"]


async def test_an_unreadable_editor_answer_changes_nothing():
    events = EventLog()
    result, _ = await run("not json at all", sink=events)
    assert result.revisions == () and result.markdown.startswith("we sold a lot [1]")
    (edited,) = [e for e in events.events if e.type == "compose.edited"]
    assert (edited.data["revised"], edited.data["rejected"]) == (0, 0)
