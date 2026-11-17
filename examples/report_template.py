"""A cited report from a template: hints, sources, formats, citation styles, editor, verifier."""

import agent_adapter as aa
from agent_adapter.compose import BlankResult, Citations, Editor, Verifier, compose_sync
from agent_adapter.retrieval import Document

sent: list[aa.LLMRequest] = []  # every request, to show how small each one is


def house_llm(request: aa.LLMRequest) -> dict:
    """A stand-in model that answers by blank; use OpenAIModel(...) or any port instead."""
    sent.append(request)
    reply = answer(request.stage.rsplit(".", 1)[-1])
    reply = {"text": reply} if isinstance(reply, str) else reply
    asked = sum(len(m.content) for m in request.messages) // 4  # about 4 characters a token
    return {**reply, "usage": {"input_tokens": asked, "output_tokens": 20}}


def answer(part: str) -> str | dict:
    if part == "summary":
        return "Revenue [grew 12% in Q3](cite:q3-results), led by Europe."
    if part == "regions":
        rows = [
            {"Region": "Europe", "Revenue": "4.1M", "Note": "[new partners](cite:eu-desk)"},
            {"Region": "Americas", "Revenue": "3.2M", "Note": "flat"},
        ]
        return {"text": "", "parsed": {"rows": rows}}
    if part == "pipeline":
        nodes = [{"id": "lead", "label": "Lead"}, {"id": "deal", "label": "Signed deal"}]
        edges = [{"source": "lead", "target": "deal", "label": "30 days"}]
        return {"text": "", "parsed": {"nodes": nodes, "edges": edges}}
    if part == "outlook":
        return "we think europe [keeps growing](cite:eu-desk) next quarter"
    if part == "aa_editor":
        fix = {"blank": "outlook", "text": "Europe should [keep growing](cite:eu-desk) in Q4."}
        return {"text": "", "parsed": {"revisions": [{**fix, "reason": "tone"}]}}
    return {"text": "", "parsed": {"issues": []}}  # the verifier: every claim is supported


model = aa.CallableModel(house_llm, provider="house", model="house-13b")
house = {"provider": "house", "free": True}
free = {"version": "house", "models": {"house-13b": house, "house-1b": house}}
ledger = aa.Ledger(prices=aa.Prices.default().merged(free))
executor = aa.Executor(aa.Registry(), ledger=ledger)

# --8<-- [start:template]
TEMPLATE = """# {{ company }}: {{ quarter }} report

{{ summary | "Summarise the quarter in two sentences." | words(40) }}

## Revenue by region

{{ regions | "Revenue by region, with a short note." | table(Region, Revenue, Note) }}

## How a deal closes

{{ pipeline | "The steps from a lead to a signed deal." | diagram(LR) }}

## Outlook

{{ outlook | "The outlook for next quarter." | words(60) | uses(summary, regions) }}

## Sources

{{ references }}
"""

SOURCES = [
    {
        "id": "q3-results",
        "text": "Q3 revenue was 7.3M, up 12% on Q2. Europe grew fastest.",
        "source": {"uri": "https://example.com/q3", "title": "Q3 results"},
    },
    {
        "id": "eu-desk",
        "text": "Three new partners in Europe signed in Q3; the pipeline doubled.",
        "source": {"uri": "https://example.com/eu", "title": "Europe desk notes"},
    },
]
# --8<-- [end:template]

# --8<-- [start:one-liner]
facts = {"company": "Acme", "quarter": "Q3"}
quick = compose_sync(TEMPLATE, model=model, inputs=facts, sources=SOURCES, executor=executor)
print(quick.text.splitlines()[2])  # Revenue grew 12% in Q3 [1], led by Europe.
# a file works too: compose_sync("report.html", ...); in async code, await compose(...)
# --8<-- [end:one-liner]


# --8<-- [start:run]
def show(blank: BlankResult) -> None:  # a live preview: a UI, a websocket, a log line
    done = "edited" if blank.edited else "filled"
    print(f"{done} {blank.name} ({blank.kind}, {'ok' if blank.ok else blank.problems})")


cached = executor.replace(cache=aa.MemoryCache())
sent.clear()
result = compose_sync(
    TEMPLATE,
    model=model,
    inputs={"company": "Acme", "quarter": "Q3"},
    sources=SOURCES,
    editor=Editor(model, style="formal, short sentences", min_words=0),  # edit a short draft
    max_usd=1.00,
    executor=cached,
    thread_id="acme-q3",  # blanks are cached per thread: a re-run reuses them
    on_blank=show,
)
print(result.text)
for revision in result.revisions:
    print(f"edited {revision.blank}: {revision.before!r} -> {revision.after!r}")
print("issues:", dict(result.issues) or "none")
# --8<-- [end:run]

# --8<-- [start:usage]
for stage, used in result.usage.items():  # blanks, editor, verifier
    print(f"{stage}: {used.calls} calls, {used.input_tokens} tokens in, ${used.cost_usd:.2f}")
size = {r.stage.rsplit(".", 1)[-1]: sum(len(m.content) for m in r.messages) for r in sent}
blank = max(n for part, n in size.items() if not part.startswith("aa_"))
print(f"prompt sizes (chars): blank at most {blank}, editor {size['aa_editor']}, ", end="")
print(f"verifier {size['aa_verify']}")
# --8<-- [end:usage]

# --8<-- [start:formats]
PAGE = """<!DOCTYPE html>
<html><body>
<h1>{{ company }}</h1>
<p>{{ summary | "Summarise the quarter." }}</p>
{{ regions | "Revenue by region." | table(Region, Revenue, Note) }}
{{ pipeline | "From lead to deal." | diagram(LR) }}
</body></html>"""
acme = {"company": "Acme & Co"}
html = compose_sync(PAGE, model=model, inputs=acme, sources=SOURCES, executor=executor)
print(html.format, html.ok, html.text.splitlines()[2])  # <h1>Acme &amp; Co</h1>

FEED = """<?xml version="1.0"?>
<report><summary>{{ summary | "Summarise the quarter." }}</summary>
<regions>{{ regions | "Revenue by region." | table(Region, Revenue, Note) }}</regions></report>"""
xml = compose_sync(FEED, model=model, sources=SOURCES, executor=executor)
print(xml.format, xml.ok)  # <cite ref="1">, <row><cell column="...">, <references>

ONE = '{{ summary | "Summarise the quarter." }}'
notes = compose_sync(ONE, model=model, format="text", sources=SOURCES, executor=executor)
print(notes.format, notes.text.splitlines()[0])
# --8<-- [end:formats]

# --8<-- [start:styles]
for style in ("numeric", "linked", "footnote", "html", "plain"):
    styled = compose_sync(ONE, model=model, sources=SOURCES, citations=style, executor=executor)
    print(f"{style}: {styled.text.splitlines()[0]}")


def superscript(words: str, source: Document, number: int) -> str:
    return f"{words}<sup>{number}</sup>"


custom = Citations(cite=superscript, references=True)  # True: keep the References list
own = compose_sync(ONE, model=model, sources=SOURCES, citations=custom, executor=executor)
print(f"custom: {own.text.splitlines()[0]}")
# --8<-- [end:styles]

checks: list[aa.LLMRequest] = []


def checker_llm(request: aa.LLMRequest) -> dict:
    """A stand-in small model for the verifier: it flags the summary once."""
    checks.append(request)
    flagged = [{"blank": "summary", "issue": "Name the quarter in words."}]
    return {"text": "", "parsed": {"issues": flagged if len(checks) == 1 else []}}


small_model = aa.CallableModel(checker_llm, provider="house", model="house-1b")

# --8<-- [start:verifier]
verifier = Verifier(model=small_model, rules=["Name the quarter."])
sent.clear()
checked = compose_sync(ONE, model=model, sources=SOURCES, verifier=verifier, executor=executor)
redone = len([r for r in sent if r.stage.endswith(".summary")])
print(f"verifier: {len(checks)} checks, summary written {redone} times, ", end="")
print(f"issues: {dict(checked.issues) or 'none'}")
# --8<-- [end:verifier]
