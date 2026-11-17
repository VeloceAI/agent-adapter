"""A report from a template: text, a table and a diagram filled by a model, then one editor."""

import agent_adapter as aa
from agent_adapter.compose import Blank, BlankResult, Editor, Template, compose


def house_llm(request: aa.LLMRequest) -> str | dict:
    """A stand-in model that answers by blank; use OpenAIModel(...) or any port instead."""
    part = request.stage.rsplit(".", 1)[-1]
    if part == "summary":
        return "Revenue grew 12% in Q3, led by Europe."
    if part == "regions":
        rows = [{"Region": "Europe", "Revenue": "4.1M"}, {"Region": "Americas", "Revenue": "3.2M"}]
        return {"text": "", "parsed": {"rows": rows}}
    if part == "pipeline":
        nodes = [{"id": "lead", "label": "Lead"}, {"id": "deal", "label": "Signed deal"}]
        edges = [{"source": "lead", "target": "deal", "label": "30 days"}]
        return {"text": "", "parsed": {"nodes": nodes, "edges": edges}}
    if part == "outlook":
        return "we think europe keeps growing next quarter"
    fix = {"blank": "outlook", "markdown": "Europe should keep growing in Q4.", "reason": "tone"}
    return {"text": "", "parsed": {"revisions": [fix]}}


model = aa.CallableModel(house_llm, provider="house", model="house-13b")

# --8<-- [start:template]
template = Template(
    """# {{ company }}: {{ quarter }} report

{{ summary }}

## Revenue by region

{{ regions }}

## How a deal closes

{{ pipeline }}

## Outlook

{{ outlook }}
""",
    [
        Blank("summary", "Summarise {{ company }}'s {{ quarter }} in two sentences.", max_words=40),
        Blank(
            "regions",
            "Revenue by region for {{ quarter }}.",
            "table",
            columns=["Region", "Revenue"],
        ),
        Blank("pipeline", "The steps from a lead to a signed deal.", "diagram", direction="LR"),
        Blank("outlook", "The outlook, given:\n{{ summary }}\n{{ regions }}", max_words=60),
    ],
    name="quarterly",
    instructions="You write for the board: plain words, no hype.",
)
# --8<-- [end:template]


# --8<-- [start:run]
def show(blank: BlankResult) -> None:  # a live preview: a UI, a websocket, a log line
    done = "edited" if blank.edited else "filled"
    print(f"{done} {blank.name} ({blank.kind}, {'ok' if blank.ok else blank.problems})")


executor = aa.Executor(aa.Registry(), cache=aa.MemoryCache())
result = aa.run_sync(
    compose(
        template,
        model=model,
        inputs={"company": "Acme", "quarter": "Q3"},
        editor=Editor(model, style="formal, short sentences"),
        executor=executor,
        thread_id="acme-q3",  # blanks are cached per thread: a re-run reuses them
        on_blank=show,
    )
)
print(result.markdown)
for revision in result.revisions:
    print(f"edited {revision.blank}: {revision.before!r} -> {revision.after!r}")
# --8<-- [end:run]
