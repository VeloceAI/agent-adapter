import pytest

from agent_adapter.compose import Blank, Citations, Template, compose
from agent_adapter.retrieval import Document, Source
from tests.compose.helpers import Script, executor, parsed, prompt_of

A = Document(
    id="a", text="Revenue grew 12% in Q3.", source=Source(uri="https://a.example", title="A")
)
B = Document(
    id="b", text="Europe led the growth.", source=Source(uri="https://b.example", title="B")
)
C = Document(id="c", text="Costs fell.", source=Source(title="C"))


async def make(text: str, blanks: list[Blank], answers: dict, **options):
    script = Script(answers)
    options.setdefault("sources", [A, B, C])
    result = await compose(
        Template(text, blanks), model=script.model, executor=executor(), **options
    )
    return result, script


async def test_two_blanks_citing_one_source_share_one_number():
    result, _ = await make(
        "{{ x }}\n\n{{ y }}",
        [Blank("x", "X"), Blank("y", "Y")],
        {"x": "Sales [grew](cite:a).", "y": "As said, [growth](cite:a) and [Europe](cite:b)."},
    )
    assert result.text == (
        "Sales grew [1].\n\nAs said, growth [1] and Europe [2].\n\n## References\n\n"
        "1. [A](https://a.example)\n2. [B](https://b.example)\n"
    )
    assert [(c.number, c.source.id, c.blank, c.text) for c in result.citations] == [
        (1, "a", "x", "grew"),
        (1, "a", "y", "growth"),
        (2, "b", "y", "Europe"),
    ]


async def test_numbers_follow_reading_order_not_fill_order():
    result, script = await make(
        "{{ late }}\n\n{{ early }}",
        [Blank("early", "E"), Blank("late", "L, after {{ early }}")],
        {"early": "[Europe](cite:b) led.", "late": "Revenue [grew](cite:a)."},
    )
    assert script.order()[:2] == ["early", "late"]
    assert result.text.startswith("Revenue grew [1].\n\nEurope [2] led.")


async def test_an_unknown_id_is_retried_once():
    result, script = await make(
        "{{ x }}", [Blank("x", "X")], {"x": ["It [grew](cite:zz).", "It [grew](cite:a)."]}
    )
    assert len(script.calls("x")) == 2
    assert "unknown sources ['zz']" in script.calls("x")[1].messages[-1].content
    assert result.text.startswith("It grew [1].") and result.ok


async def test_a_citation_in_a_table_cell_and_unused_sources_dropped():
    rows = {"rows": [{"Region": "EU", "Note": "[led](cite:b)"}]}
    result, _ = await make(
        "{{ t }}", [Blank("t", "T", "table", columns=["Region", "Note"])], {"t": parsed(rows)}
    )
    assert "| EU | led [1] |" in result.text
    assert "1. [B](https://b.example)" in result.text and "A" not in result.text.split("##")[1]
    assert [c.source.id for c in result.citations] == ["b"]


async def test_old_markers_a_references_slot_and_dedupe_by_url():
    twin = Document(id="a2", text="Same page.", source=Source(uri="https://a.example"))
    result, _ = await make(
        "{{ x }}\n\nSee:\n{{ references }}\nEnd.",
        [Blank("x", "X")],
        {"x": "Revenue grew. [[a]] Same. [[a2]] Costs fell [here](cite:c)."},
        sources=[A, twin, C],
    )
    assert result.text == (
        "Revenue grew. [1] Same. [1] Costs fell here [2].\n\nSee:\n"
        "1. [A](https://a.example)\n2. C\nEnd."
    )
    assert result.citations[0].text == "Revenue grew."


async def test_each_blank_sees_only_its_best_passages():
    many = [Document(id=f"d{i}", text=f"Note {i} about weather.") for i in range(8)]
    sales = Document(id="s", text="Sales revenue grew strongly.")
    _, script = await make(
        "{{ x }}", [Blank("x", "Sales revenue")], {"x": "ok"}, sources=[*many, sales], top_k=2
    )
    sent = prompt_of(script.calls("x")[0])
    assert sent.splitlines()[2:] == ["Sources:", "[s] Sales revenue grew strongly."]
    _, script = await make("{{ x }}", [Blank("x", "Weather")], {"x": "ok"}, sources=many, top_k=2)
    assert prompt_of(script.calls("x")[0]).count("\n[d") == 2
    own = Blank("y", "Y", sources=[C])
    _, script = await make("{{ y }}", [own], {"y": "ok"})
    assert "[c] Costs fell." in prompt_of(script.calls("y")[0])
    assert "[a]" not in prompt_of(script.calls("y")[0])


STYLE_CASES = {
    "numeric": ("Sales grew [1].", "1. [A](https://a.example)"),
    "linked": ("Sales [grew](https://a.example).", None),
    "footnote": ("Sales grew[^1].", "[^1]: [A](https://a.example)"),
    "html": ('Sales <a class="aa-cite" href="https://a.example" title="A">grew</a>.', None),
    "plain": ("Sales grew.", None),
}


@pytest.mark.parametrize("style", list(STYLE_CASES))
async def test_each_renderer(style):
    result, _ = await make(
        "{{ x }}", [Blank("x", "X")], {"x": "Sales [grew](cite:a)."}, citations=style
    )
    body, refs = STYLE_CASES[style]
    assert result.text.startswith(body)
    if refs is None:
        assert result.text.rstrip() == body
    else:
        assert result.text.rstrip().endswith(refs)


async def test_a_custom_renderer_and_references_writer():
    style = Citations(
        cite=lambda words, source, n: f"{words}<{n}:{source.id}>",
        references=lambda pairs: "Sources: " + ", ".join(d.source.title for _, d in pairs),
    )
    result, _ = await make(
        "{{ x }}", [Blank("x", "X")], {"x": "[Up](cite:a), [down](cite:c)."}, citations=style
    )
    assert result.text == "Up<1:a>, down<2:c>.\n\n## References\n\nSources: A, C\n"


async def test_a_bad_style_is_refused():
    with pytest.raises(ValueError, match="citations must be one of"):
        await make("{{ x }}", [Blank("x", "X")], {"x": "x"}, citations="apa")
