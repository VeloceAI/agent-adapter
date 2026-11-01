# WP04 Cost and tracing sinks — TDD plan

Contract: `docs/superpowers/specs/2026-10-29-interface-contract.md` rev 2 (§9.1, §9.2, §9.4,
§12.2, §12.3, §21.3 WP04). Research: R4 §5, §6, §8. Branch `wp/04-cost-tracing`.
Consumes WP01 only (§3, §5, §12.1). Every task: failing test first, then code, then gates.

## Tasks

| # | Task | Files | Tests |
|---|---|---|---|
| 1 | Price table parsing: flat prices, tiers, periods, caps-only, aliases, units, strict keys, `ConfigError` on every invalid shape | `cost/prices.py` | `tests/cost/test_prices_load.py` |
| 2 | `resolve` (exact, alias, longest prefix), `price` (tier by prompt tokens, period by date), `cost` formula incl. TTL writes and units, `unit_cost` incl. `<provider>:<model>:cache_storage`, `caps`, `merged`, unknown model warns once | `cost/prices.py` | `tests/cost/test_prices.py` |
| 3 | Shipped table: R4 §6 OpenAI + Gemini rows, Anthropic caps-only, no `gpt-5.6` write prices, no `gemini-3-pro-preview`, no Voyage | `cost/prices_default.json` | `tests/cost/test_prices_default.py` |
| 4 | Ledger sinks: `MemorySink`, `CallbackSink`, `JsonlSink`, `read_jsonl` (last write wins, torn last line) | `cost/sinks.py` | `tests/cost/test_sinks.py` |
| 5 | `Ledger`: pricing on record (llm / api / cache hit), idempotent replace, `max_runs` eviction, filters, sink failure logged, never raises | `cost/ledger.py` | `tests/cost/test_ledger.py` |
| 6 | `summary`: by stage / kind, hit rates, wall / busy / parallelism | `cost/summary.py` | `tests/cost/test_summary.py` |
| 7 | Event sinks: `EventLog` (bounded), `LoggingSink`, `CallbackEventSink`, `FanoutSink` | `tracing/sinks.py`, `tracing/__init__.py` | `tests/tracing/test_sinks.py` |
| 8 | `semconv.py` constants (checked against `opentelemetry-semantic-conventions` when installed) | `tracing/semconv.py` | `tests/tracing/test_semconv.py` |
| 9 | `OTelSink`: llm / tool / flow / node spans, kinds, attributes, parenting, error status, metrics, content gate; `MissingDependency` without the extra | `tracing/otel.py` | `tests/tracing/test_otel.py` (in-memory exporter; `importorskip`) |
| 10 | Docs + examples: `concepts/cost.md`, `concepts/tracing.md`, `examples/cost.py`, `examples/tracing.py`; SHARED-EDITS (nav, features rows, CHANGELOG) | `docs/**`, `examples/*`, `tests/test_examples_wp04.py`, `SHARED-EDITS.md` | `tests/test_examples_wp04.py`, `mkdocs build --strict` |

## Exit gate

- `pytest -q` green, 0 warnings; ≥ 90 % line + branch coverage on `agent_adapter/cost/*`,
  `tracing/sinks.py`, `tracing/semconv.py`; `tracing/otel.py` ≥ 90 % in `nox -s models`.
- `ruff check .`, `ruff format --check .`, `mypy` strict, `mkdocs build --strict`,
  `tests/test_boundaries.py` green; no file over 500 lines.
