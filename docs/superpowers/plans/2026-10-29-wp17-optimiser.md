# WP17 — Optimiser: TDD plan

Contract: `docs/superpowers/specs/2026-10-29-interface-contract.md` §17 (17.1–17.5), §20.3
(Optimiser row), §21.3 WP17. Consumes §8 (executor), §9.4 (ledger), §10 (cache), §11.3
(`InstrumentedModel`), §13 (flows). Intent: spec rev 2 §15; internals per R1 Part B.
Rulings: `proposer_calls == 2k` (retries in `Candidate.attempts`); merge, reflection and
selection calls counted separately (I17); `harness.after.json` snapshot (R1 B8).

## Tasks

| # | Task | Files | Tests first |
|---|---|---|---|
| 1 | Plan | this file | — |
| 2 | Data model | `optimiser/model.py` | `tests/optimiser/test_model.py`: `with_edits` (set, delete, inherit), `diff`, `unified_diff` (JSON pretty-printed, headers), frozen models, `Score` range |
| 3 | Surface | `optimiser/surface.py`, `optimiser/_schema_check.py` | `test_surface.py`: path grammar (each kind, escapes, case), caps per pattern, dimension check, hook-config JSON + schema, host problems, `edits_between` round trip |
| 4 | Hosts | `optimiser/hosts.py`, `optimiser/_transcript.py` | `test_hosts.py`: `apply_tree` copy, `RegistryHost` surface / caps / validate / run (tree applied in memory, hints rendered, hook config applied, errors → `CaseRun(ok=False)`, control flow propagates), `FlowHost` (node prompts via `with_prompts`, `node_model`, transcript recorder) |
| 5 | Evidence + prompts | `optimiser/evidence.py`, `optimiser/prompts.py` | `test_evidence.py`: bounded by `evidence_chars`, observations clipped at `obs_trunc`, worst case first, scores hidden unless `show_scores_to_proposer`, redaction; `test_prompts.py`: per-dimension text, schemas portable, no forbidden words |
| 6 | Propose | `optimiser/propose.py` | `test_propose.py`: valid proposal, KEEP_BASE, off-dimension / bad path / over cap → retry with feedback → fall back; invalid JSON retried; `ModelError` falls back; attempts counted; budget error propagates |
| 7 | Probe | `optimiser/probe.py` | `test_probe.py`: every case run once, judge failure → `None`, host exception → failed run, cost per case from the ledger, contrasts (`edit_exercised`, deltas, bounded diff), second base probe free through the cache |
| 8 | Compose + select | `optimiser/compose.py`, `optimiser/_select.py` | `test_compose.py`: admission (gain, unsafe never admitted, unexercised, KEEP_BASE / fell back), conflicts greedy and llm (and llm failure → greedy), decision lines; `test_select.py`: judge ranking, ties, KEEP_BASE, blind selection |
| 9 | Lessons | `optimiser/lessons.py` | `test_lessons.py`: atomic store, filter by domain / paths / limit, NO_ENTRY not stored, confounded-only dropped, corrupt file → `OptimiserError` |
| 10 | Loop + resume | `optimiser/loop.py`, `optimiser/_state.py` | `test_loop.py`: parallel and sequential identity (`(1+3k)·N` case runs, `2k` proposer calls), KEEP_BASE, budget stops (usd via `BudgetHook`, case runs, wall) write a partial report, resume restores finished stages without model calls |
| 11 | Report | `optimiser/report.py` | `test_report.py`: `harness.patch`, `report.md`, `result.json` (transcripts by path), `runs/<label>/<case>.md`, `harness.after.json`, atomic writes, no secrets |
| 12 | CLI | `optimiser/__main__.py` | `test_cli.py`: `--dry-run` (surface, cases, identity, no model call), exit 0 / 2 / 1, `provider:model` parsing |
| 13 | Exports | `optimiser/__init__.py` | `test_package.py` |
| 14 | Docs + example | `docs/optimiser.md`, `examples/optimiser_demo.py`, `tests/optimiser/test_examples_optimiser.py` | example runs offline; `mkdocs build --strict` |
| 15 | Shared edits | `SHARED-EDITS.md` | — |

Shared fakes: `tests/optimiser/fakes.py` — a toy `FakeHost`, `ScriptedProposer` (answers by
stage label), exact-match judge, priced fake replies.

## Exit gate

`pytest -q` (0 warnings, ≥ 90 % line + branch on `agent_adapter/optimiser`), `ruff check .`,
`ruff format --check .`, `mypy` (strict), `mkdocs build --strict`, boundary test, no file
over 500 lines; every Optimiser row of §20.3; `--dry-run` prints surface, cases and the
budget identity with no model call; prompts written fresh, harness2 cited in docs only.
