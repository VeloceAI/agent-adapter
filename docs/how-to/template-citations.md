---
title: Citations and token use in template documents
description: Cite sources in a template document with numbered, linked, footnote, HTML or custom citations, and keep each model call small with passages, a lean editor and verifier, usage per stage and a spend cap.
---

# Citations and token use

A [template document](template-documents.md) can cite its sources. Each blank sees only
the few passages that match it, the code numbers what it cites, and `result.usage` shows
what each stage cost. Snippets come from `examples/report_template.py`.

## Sources and markers

Give `sources=` (`Document`s, strings or mappings with `id`, `text` and `source`; see
[Retrieval](../concepts/retrieval.md)), or a `retriever=`. A blank may have its own
`Blank(..., sources=[...])`, which replace the document's for that blank.

A blank cites by wrapping the words a source supports: `[grew 12%](cite:q3-results)`.
The older `[[q3-results]]` also works and backs the sentence before it. Markers work in
text, list items, table cells and diagram labels. An id the blank was not shown fails its
checks, so the blank is asked once more.

## Numbering

After the blanks, the editor and the verifier finish, the code numbers sources by first
appearance in reading order (the order in the template, not the fill order). Two blanks
citing one source share its number, two ids with the same URL share one too, and sources
nobody cited are dropped. The References list goes into `{{ references }}`, or at the
end: under a heading in Markdown, a `<section>` before `</body>` in HTML, a
`<references>` element in XML. `result.citations` lists each `Citation(number, source,
blank, text)`.

## Styles

```python
--8<-- "examples/report_template.py:styles"
```

| `citations=` | Cited words become | References list |
|---|---|---|
| `"numeric"` (default) | `grew 12% [1]` | yes |
| `"linked"` | `[grew 12%](https://...)` | no |
| `"footnote"` | `grew 12%[^1]` | footnote definitions |
| `"html"` | `<a class="aa-cite" href="..." title="...">grew 12%</a>` | no |
| `"plain"` | `grew 12%` | no |
| a function `(words, source, number) -> str` | your text | `Citations(references=)` |

`Citations(style=, cite=, references=)` sets both parts: `references=True` or `False`
forces the list, and a function `(pairs) -> str` writes it from `(number, source)` pairs.
Each style is drawn for the template's format: in XML, numbers become
`<cite ref="1">words</cite>`.

## Few tokens

```python
--8<-- "examples/report_template.py:usage"
```

- **Blanks**: a prompt holds the blank's hint, the blanks it uses and its best `top_k=4`
  passages of about 600 characters, ranked by BM25 against the hint, or picked by your
  `retriever=`. It never holds the whole document or every source.
- **Editor**: only text blanks, as JSON `id: text`; it returns only what it changes and is
  skipped with one text blank or under `min_words=150` words.
- **Verifier**: only the cited phrases, each with about 300 characters of its source, in
  one call. `Verifier(model=small_model)` runs it on a cheaper model.
- **Spend**: `max_usd=` adds a `BudgetHook`, so the run stops with `BudgetExceeded` once
  it costs that much. Blanks are cached (see the cache section of the main page).

`result.usage` maps `blanks`, `editor` and `verifier` to `StageUsage(calls,
input_tokens, output_tokens, cost_usd)`, read from the executor's ledger (one is added
when the executor has none). In a 6-blank template with 10 sources (28,000 characters),
each blank prompt stays under 2,700 characters, the editor's under 1,700 and the
verifier's under 1,800; a test keeps it so.

See also: [Template documents](template-documents.md), [Retrieval](../concepts/retrieval.md),
[Cost and budgets](../concepts/cost.md).
