# WP10 — LangChain, genai AFC, OpenAPI importer · TDD plan

Branch `wp/10-langchain-genai-openapi` from `main` 42ee50a. Contract rev 3 §4.3, §8, §14.1–§14.4.

## Tasks

| # | Task | Files | Tests first |
|---|---|---|---|
| 1 | Demo plugin: `executor=None` uses `runner_for` (resolves `TODO(WP10)`) | `tests/fixtures/demo_integration/demo_integration/__init__.py` | `test_kit_selftest.py`: default executor runs the demo tool |
| 2 | genai AFC proxies: `afc_callable`, `afc_callables` (synthesised `__signature__`/`__doc__`, schema-exact annotations, non-identifier names) | `integrations/gemini.py` | `test_gemini_afc.py` (no genai needed), AFC harness in `test_gemini_contract.py` (real `generate_content` over `httpx.MockTransport`) |
| 3 | `run_function_calls(calls, *, executor=None, ctx=None, registry=None)` → `types.Part`s; executor path concurrent via `Executor.parallel`, other runners sequential | `integrations/gemini.py` | function-call harness + unit tests in `test_gemini_contract.py` |
| 4 | LangChain export: `structured_tool`, `structured_tools`, native call id via `arun(tool_call_id=…)`, `ToolException` channel, sync `invoke` via `run_sync` | `integrations/langchain.py` | kit C1–C8 in `test_langchain_contract.py` |
| 5 | LangChain import: `import_langchain` (schema via `convert_to_openai_function`, `InjectedToolArg` out, name sanitising, `ToolException` → `ToolError`, `ValidationError` → `ToolArgsInvalid`) | `integrations/langchain.py` | `test_langchain_contract.py` |
| 6 | OpenAPI importer: document loading (Mapping / JSON / YAML / path), `$ref` inlining with cycle guard, flat input schema (path/query/header/cookie + JSON body, `body_` clash prefix, non-object `body`), sanitised names, auth types, status mapping, `LoopLocal` client | `integrations/openapi/{__init__,_document,_schema,_http,auth}.py` | `test_openapi_contract.py` + `openapi_cases*.py` over `httpx.MockTransport` |
| 7 | Contract files for the renderer integrations (host-dispatch harnesses) | `tests/integrations/test_{openai,anthropic,gemini}_contract.py` | kit C1–C8 |
| 8 | Kit fixes found while building the harnesses | `tests/integrations/kit.py` | `test_kit_selftest.py` |
| 9 | Docs + examples | `docs/integrations/{langchain,gemini,openapi}.md`, `examples/{langchain_tools,gemini_afc,openapi_import}.py`, `tests/integrations/test_wp10_examples.py` | example tests |
| 10 | SHARED-EDITS | `SHARED-EDITS.md` | — |

## Exit gate

- Kit C1–C8 green for `langchain`, `gemini` (both paths), `openai`, `anthropic`, demo.
- `import_langchain` keeps `InjectedToolArg` out of the schema.
- OpenAPI over `httpx.MockTransport`: path/query/header/cookie/body, sanitised names, `$ref`,
  YAML, Bearer/Header/Query auth, 4xx → `ToolError`, 429/5xx retried then `ToolFailed`.
- `pytest -q` 0 warnings; `ruff check`, `ruff format --check`, `mypy` strict, `mkdocs build --strict`;
  nox `contract-langchain`, `contract-gemini`, `contract-openapi`, `contract-openai`,
  `contract-anthropic`, `plugin` (each ≥ 90 % on its module).
