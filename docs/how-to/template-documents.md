---
title: Template documents filled by agents
description: Build one report from many model or agent calls with a Markdown, HTML, XML or text template, typed blanks, checks, retries, an editor, a verifier and a per-blank cache.
---

# Template documents

Write the document once with `{{ name }}` blanks, in Markdown, HTML, XML or plain text.
Each blank is filled by a model or an agent, independent blanks in parallel. Tables and
diagrams come back as data and the code draws them, so the document stays well formed.
An editor makes the wording fit, a verifier checks the result, and sources are cited
([Citations and token use](template-citations.md)).

Every snippet on this page comes from `examples/report_template.py`, which runs offline on
a stand-in model.

## One line

```python
--8<-- "examples/report_template.py:one-liner"
```

`compose()` is the async form. The template can be a file path (`report.html`,
`report.xml`, `report.md`, `notes.txt`), template text, or a `Template`.

## The template

```python
--8<-- "examples/report_template.py:template"
```

A slot with a hint is a blank: `{{ name | "what to write" | option ... }}`. Any other
`{{ name }}` (here `company` and `quarter`) is an input you pass in `inputs=`.

| Option | Means |
|---|---|
| `"two sentences on the outcome"` | the blank's prompt |
| `list`, `number`, `table(A, B)`, `diagram(LR)` | its kind (`text` by default) |
| `words(40)` | at most 40 words |
| `optional` | may stay empty |
| `uses(summary, regions)` | fills after those blanks and sees their text |

For more control, pass `Blank(name, prompt, kind=, agent=, max_words=, sources=)` objects
as `blanks=`; a prompt may then use `{{ other_blank }}` directly. A cycle, a bad name or a
bad option is refused when the template is built, with every problem listed together.
`{{ references }}` marks where the References list goes. `\{{` keeps literal braces.

## Formats

```python
--8<-- "examples/report_template.py:formats"
```

The format comes from the file extension or the content, or `format=`. Inputs and model
text are escaped for it, and blanks are told to write plain text for HTML, XML and text.
Lists, tables and diagrams are drawn per format: a pipe table or `<table>`, a Mermaid
fence or `<pre class="mermaid">`, and `<item>`, `<row>`, `<cell>`, `<node>` and `<edge>`
elements in XML.

## A full run

```python
--8<-- "examples/report_template.py:run"
```

| `ComposeResult` field | What it holds |
|---|---|
| `.text` (`.markdown`) | the finished document, in the template's format |
| `.blanks` | `{name: BlankResult}`: `value`, `problems`, `attempts`, `cached`, `edited`, `cost_usd` |
| `.citations` | `Citation(number, source, blank, text)` for every cited phrase |
| `.revisions` | every editor revision, accepted or not, with `before` and `after` |
| `.issues` / `.ok` | what is still wrong, by blank (`"document"` for the whole text) |
| `.usage` / `.cost` | calls, tokens and USD for `blanks`, `editor` and `verifier` |

## Checks and the single retry

Each blank is checked as it fills: not empty (unless `optional`), within `words`, valid
JSON of the right shape, table rows with exactly the columns, diagram edges between known
nodes, and only the source ids it was shown. A failing blank is asked once more, alone,
with its answer and the problems. If it fails again it is kept and listed in `issues`;
the blanks after it still run.

## The editor

On by default with the document's model; `Editor(model, style="formal")` sets the voice
and `editor=False` turns it off. It reads only the text blanks and returns only the ones
it changes; with one text blank, or under `min_words=150` words, it is skipped. Tables,
lists and diagrams stay exactly as filled. A revision that changes which sources a blank
cites, or fails its checks, is rejected and recorded with `accepted=False` and its
`problems`.

## The verifier

```python
--8<-- "examples/report_template.py:verifier"
```

On by default. Code checks always run: the document is well formed (XML parses, HTML tags
close, Markdown tables keep their shape), every blank is filled, every cited id is known
and the reference numbers match. With sources, the document's model also checks each
cited phrase against its source; `Verifier(model=small_model, rules=[...])` uses another
model and adds rules. A flagged blank is written again once, alone, with the issue; what
is left goes to `result.issues`. `verifier=False` keeps only the code checks.

## Per-blank agents

```python
researcher = LoopAgent(model, registry.subset(names=["search_sales"]), instructions="Cite numbers.")
Blank("regions", "Revenue by region.", "table", columns=["Region", "Revenue"], agent=researcher)
```

Its tool calls run through the executor, so approvals, budgets, guards and the ledger
apply. Give the executor the agent's registry (`aa.Executor(registry)`).

## Cache, preview and events

- With a cache on the executor and a `thread_id`, each blank is cached by what it depends
  on: its settings, its model, the blanks and inputs it uses, and its sources. Change one
  blank and only it, the blanks that use it, the editor and the verifier run again.
  `cache_scope="global"` is for public data only; `None` turns caching off.
- `on_blank(result)` (sync or async) gets each blank as it fills, and again after an
  edit or a redo. A failing callback is logged, never raised.
- Events: `compose.blank_started`, `compose.blank_filled` and `compose.edited`. Texts go
  only to sinks that capture content. `agent-adapter ui` shows the document filling in.

## Things to know

- The template compiles to a `Flow` (`aa_begin`, a node per blank, `aa_editor`,
  `aa_verify`). `concurrency=4` caps blanks in flight; `runner=FlowRunner(executor,
  checkpoints=...)` adds checkpoints. Pass `ctx=` to compose inside a running tool.
- A blank redone by the verifier does not re-run the blanks that used its earlier text.

See also: [Citations and token use](template-citations.md), [Flows](../concepts/flows.md),
[Multi-agent](multi-agent.md), [Caching](caching.md), [See your runs](see-your-runs.md).
