import xml.etree.ElementTree as ET

import pytest

from agent_adapter.compose import Blank, Template, compose, compose_sync
from agent_adapter.compose.formats import detect, problems, render
from agent_adapter.compose.hints import parse
from agent_adapter.errors import FlowDefinitionError
from tests.compose.helpers import Script, executor, parsed

SOURCE = {"id": "a", "text": "Sales rose.", "source": {"uri": "https://a.example", "title": "A"}}
ROWS = {"rows": [{"Region": "EU & UK", "Note": "<up>"}]}
GRAPH = {
    "nodes": [{"id": "x", "label": "Lead"}, {"id": "y", "label": "Deal"}],
    "edges": [{"source": "x", "target": "y", "label": ""}],
}
ANSWERS = {"intro": "Sales <b>[rose](cite:a)</b> & more.", "t": parsed(ROWS), "d": parsed(GRAPH)}


def test_formats_are_detected_from_the_extension_or_the_content():
    assert detect("", "r.HTML") == "html" and detect("", "r.xml") == "xml"
    assert detect("", "notes.txt") == "text" and detect("# Hi", "r.md") == "markdown"
    assert detect('<?xml version="1.0"?><r/>') == "xml"
    assert detect("<!DOCTYPE html><html></html>") == "html"
    assert detect("<div>{{ a }}</div>") == "html" and detect("# {{ a }}") == "markdown"


def test_hints_make_blanks_without_blank_objects():
    hint = parse(' rows | "Sales by region" | table(Region, Sales) | words(40) | optional ')
    assert (hint.name, hint.hint, hint.kind, hint.columns) == (
        "rows",
        "Sales by region",
        "table",
        ("Region", "Sales"),
    )
    assert hint.max_words == 40 and not hint.required and hint.marked
    assert parse("plain").marked is False
    template = Template(
        '{{ summary | "two sentences" | words(30) }}\n{{ flow | diagram(lr) }}\n'
        '{{ verdict | "a verdict" | uses(summary) }} {{ who }}'
    )
    assert set(template.blanks) == {"summary", "flow", "verdict"}
    assert template.inputs == ("who",)
    assert template.blanks["flow"].direction == "LR"
    assert template.layers() == [["summary", "flow"], ["verdict"]]
    assert template.blanks["verdict"].prompt == "a verdict\n\nBuild on:\n{{ summary }}"


def test_bad_hints_are_reported():
    with pytest.raises(FlowDefinitionError) as caught:
        Template('{{ a | "x" | colour(red) | words(many) }}')
    text = "\n".join(caught.value.problems)
    assert "unknown option 'colour(red)'" in text and "unknown option 'words(many)'" in text


async def test_html_escapes_model_text_and_draws_tables_and_diagrams():
    page = (
        '<!DOCTYPE html><html><body><h1>{{ title }}</h1>{{ intro | "Intro" }}'
        "{{ t | table(Region, Note) }}{{ d | diagram }}</body></html>"
    )
    script = Script(ANSWERS)
    result = await compose(
        page,
        model=script.model,
        inputs={"title": "Q3 <draft>"},
        sources=[SOURCE],
        executor=executor(),
    )
    assert result.format == "html" and result.ok, result.issues
    assert "<h1>Q3 &lt;draft&gt;</h1>" in result.text
    assert "Sales &lt;b&gt;rose [1]&lt;/b&gt; &amp; more." in result.text
    assert "<td>EU &amp; UK</td><td>&lt;up&gt;</td>" in result.text
    assert '<pre class="mermaid">' in result.text and "n1 --&gt; n2" in result.text
    assert '<ol class="aa-references"><li id="aa-ref-1"><a href="https://a.example">A</a>' in (
        result.text
    )
    assert result.text.endswith("</ol></section></body></html>")
    system = script.calls("intro")[0].messages[0].content
    assert "plain text with no markup" in system


async def test_xml_stays_well_formed():
    doc = (
        '<?xml version="1.0"?>\n<report><intro>{{ intro | "Intro" }}</intro>'
        "<t>{{ t | table(Region, Note) }}</t><d>{{ d | diagram }}</d></report>"
    )
    result = await compose(doc, model=Script(ANSWERS).model, sources=[SOURCE], executor=executor())
    assert result.format == "xml" and result.ok, result.issues
    root = ET.fromstring(result.text.encode())
    assert root.find("intro/cite").text == "rose"
    assert root.find("t/row/cell").attrib == {"column": "Region"}
    assert [n.text for n in root.find("d/diagram")][:2] == ["Lead", "Deal"]
    assert root.find("references/source").attrib["uri"] == "https://a.example"


async def test_text_and_markdown_output():
    script = Script(ANSWERS)
    text = await compose(
        '{{ intro | "Intro" }}\n{{ t | table(Region, Note) }}',
        model=script.model,
        format="text",
        sources=[SOURCE],
        executor=executor(),
    )
    assert text.text.startswith("Sales <b>rose [1]</b> & more.\nRegion | Note\nEU & UK | <up>")
    assert text.text.endswith("References\n[1] A - https://a.example\n")
    md = await compose(
        '{{ intro | "Intro" }}', model=Script(ANSWERS).model, sources=[SOURCE], executor=executor()
    )
    assert md.text.startswith("Sales <b>rose [1]</b> & more.")


def test_well_formed_checks():
    assert problems("<r><a></r>", "xml")[0].startswith("the XML is not well formed")
    assert problems("<div><span></div>", "html") == [
        "</div> closes nothing open",
        "<div> is never closed",
        "<span> is never closed",
    ]
    assert problems("<p>one<p>two<br></div>", "html") == ["</div> closes nothing open"]
    assert problems("| a | b |\n| --- | --- |\n| 1 |\n", "markdown")
    assert problems("```\n| a |\n| a | b |\n```", "markdown") == []
    assert render(Blank("n", "x", "number"), 2.0, "html") == "2"
    assert render(Blank("l", "x", "list"), ["<a>"], "html") == "<ul><li>&lt;a&gt;</li></ul>"


async def test_the_one_liner_reads_a_file_and_compose_sync_works(tmp_path):
    path = tmp_path / "brief.html"
    path.write_text('<html><body><p>{{ intro | "Intro" }}</p></body></html>', encoding="utf-8")
    result = await compose(str(path), model=Script(ANSWERS).model, sources=[SOURCE])
    assert result.format == "html" and "rose [1]" in result.text
    assert result.ok, result.issues


def test_compose_sync(tmp_path):
    path = tmp_path / "brief.md"
    path.write_text('# Brief\n\n{{ intro | "Intro" }}\n', encoding="utf-8")
    result = compose_sync(path, model=Script(ANSWERS).model, sources=[SOURCE])
    assert result.text.startswith("# Brief\n\nSales <b>rose [1]</b> & more.")
    assert result.usage["blanks"].calls == 1 and result.usage["verifier"].calls == 1
