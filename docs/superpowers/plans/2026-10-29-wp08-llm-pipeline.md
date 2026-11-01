# WP08 — LLM pipeline (plan)

Contract rev 3: §11.3 `InstrumentedModel`, §11.4 `run_tool_loop`, §8.2 `Executor.model()`,
§10.6 default plug-in caps, §12.1 llm events, I7 / I8 rulings.

## Tasks (test first, one commit per group)

| # | Task | Files | Tests |
|---|---|---|---|
| 1 | Plan | this file | — |
| 2 | Default plug-in caps → `Prices.default().caps` | `cache/providers/_base.py`, plug-in docstrings | `tests/cache/providers/test_default_caps.py` |
| 3 | `InstrumentedModel.generate`: ctx resolution (bound / current / fresh, I7), entry ids with `aa.llm_seq`, hooks, response cache, ledger, events, `use_context` around the port | `models/instrument.py`, `models/_instrument_call.py` (if > 400 lines), `models/__init__.py` | `tests/models/test_instrument.py` |
| 4 | `InstrumentedModel.stream`: same around the stream, no response cache, per-step context | same | `tests/models/test_instrument_stream.py` |
| 5 | `Executor.model()` | `exec/executor.py` | `tests/exec/test_executor_model.py` |
| 6 | `run_tool_loop` + `ToolLoopResult`; stop on length / content_filter / pending / max_steps; resume of a trailing tool-call turn | `exec/tool_loop.py`, `exec/__init__.py` | `tests/exec/test_tool_loop.py` |
| 7 | Exit-gate end-to-end: ScriptedModel + Ledger + MemoryCache + EventLog (second loop costs 0, deterministic ids, Fallback I8 both ways, pending stops) | — | `tests/exec/test_tool_loop_e2e.py` |
| 8 | Real plug-ins on the wire with default caps (`prompt_cache_key`, `prompt_cache_options`, `cachedContent`, `cache_control`) | — | `tests/models/test_plugins_on_the_wire.py` (models session; skipped without SDKs) |
| 9 | Docs + example | `docs/concepts/tool-loop.md`, rewrite `docs/architecture/hitl-cost-cache.md`, `examples/tool_loop.py`, `tests/test_examples_wp08.py`, caps note in `docs/concepts/cache.md`, file row in `docs/concepts/models.md` | example test |
| 10 | SHARED-EDITS | `SHARED-EDITS.md` (exports, nav, features rows, CHANGELOG) | — |

## Exit gate

- pytest 0 warnings, new modules ≥ 90 % line + branch; ruff, ruff format, mypy strict, mkdocs strict.
- nox `models` session (wire tests) green with `--envdir D:\aa\nox-wp08`.
- Boundary test passes (`exec → models`, `cache → cost`, `models → cost` only).
- No file over 500 lines.
