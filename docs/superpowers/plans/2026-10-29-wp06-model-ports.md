# WP06 Model ports — TDD plan

Contract: `docs/superpowers/specs/2026-10-29-interface-contract.md` rev 2 (§3.3, §3.4, §9.3,
§11.1, §11.2, §20.1–§20.3, §21.3 WP06). Research: R4 §1–§5, §10, §11.
Branch `wp/06-models`. Every task: failing test first, then code, then gates.

## Layout

SDK-free code (dict in, dict out) lives in private modules so the core session covers it
without any SDK; the three ODMs hold only the port classes, SDK calls and error maps and
re-export the contract's build/parse functions.

| File | Kind | Holds |
|---|---|---|
| `models/_convert.py` | pure | dict/attr access, JSON args, structured-output parsing, effort map, kwargs merge |
| `models/errors.py` | pure | status → `ModelError` class, `Retry-After` parsing, `model_error()` builder |
| `models/usage.py` | pure | §9.3 normalisers |
| `models/_port.py` | pure | `ProviderPort` base: loop-local clients, plug-in, generate-via-stream, error wrap |
| `models/_openai_chat.py`, `_openai_responses.py` | pure | build / parse / stream assembly |
| `models/_gemini_convert.py`, `_anthropic_convert.py` | pure | build / parse / stream assembly |
| `models/openai.py`, `gemini.py`, `anthropic.py` | ODM | port class, SDK calls, `map_*_error` |
| `models/callable.py`, `fallback.py` | pure | `CallableModel`, `FallbackModel` |
| `tests/fakes/models.py` | fake | `ScriptedModel`, `text()`, `calls()`, `FakePromptCachePlugin` |

## Tasks

| # | Task | Tests |
|---|---|---|
| 1 | §23 #2 evidence: OpenAI cache-write accounting (pricing/caching page + SDK docstrings) | recorded in `docs/concepts/models.md` |
| 2 | `usage.py`: every §9.3 cell, dict + SDK object + full response, missing → 0, TTL split | `tests/models/test_usage.py`, `test_usage_sdk.py` |
| 3 | `errors.py` helpers + `_convert.py` | `tests/models/test_errors.py`, `test_convert.py` |
| 4 | OpenAI Chat: build (tools strict, choice, schema, reasoning, stop, seed), parse (text, calls, refusal, finish map, usage), stream assembly | `tests/models/test_openai_chat.py` |
| 5 | OpenAI Responses: build (instructions-free input items, raw echo, flat tools, `text.format`), parse, stream by `item_id` | `tests/models/test_openai_responses.py` |
| 6 | `ProviderPort` + `OpenAIModel`: transport path (httpx2 MockTransport), fake plug-in kwargs on the wire, defaults/extra order, TTL fill, error map rows, `generate(stream=True)`, loop-local clients over two `run_sync` | `tests/models/test_openai_port.py`, `test_openai_errors.py` |
| 7 | Gemini: build (system, raw parts echo, function responses `output`/`error`, ids, AFC off, tool_config, `response_json_schema`, thinking), parse (finish map, block reason, thought parts), stream; port via httpx MockTransport; error map | `tests/models/test_gemini_convert.py`, `test_gemini_port.py` |
| 8 | Anthropic: build (system blocks, tool_use/tool_result, `is_error`, tool_choice, `output_config`), parse, stream (`input_json_delta`); port + error map | `tests/models/test_anthropic_convert.py`, `test_anthropic_port.py` |
| 9 | `CallableModel`: str / mapping / response returns, structured output, stream_fn, synthesised stream | `tests/models/test_callable.py` |
| 10 | `FallbackModel`: fall_on, other errors propagate, last error, `llm.fallback` event, stream before first event only | `tests/models/test_fallback.py` |
| 11 | Fakes: `ScriptedModel`, `text`, `calls`, `FakePromptCachePlugin` | `tests/fakes/test_model_fakes.py` |
| 12 | Docs `docs/concepts/models.md` (+ mermaid), `examples/models.py` + `tests/test_examples.py`, SHARED-EDITS (nav, features rows, CHANGELOG, exports) | `mkdocs build --strict` |

## Exit gate

- `pytest -q`: 0 warnings; core coverage ≥ 90 % (pure modules); ODMs ≥ 90 % in a `models`-style
  run (`--cov=agent_adapter.models.openai` etc. with the ODM config).
- `ruff check .`, `ruff format --check .`, `mypy` strict, `tests/test_boundaries.py`.
- No file over 500 lines; docs page ≤ 150 lines.
