# WP01 Foundation — TDD plan

Contract: `docs/superpowers/specs/2026-10-29-interface-contract.md` (§2.3, §3, §4, §5, §6.1,
§10.1, §10.2, §10.4 `ToolResultCache`, §12.1, §18, §20.2, §21.3 WP01).
Branch `wp/01-foundation`. Every task: failing test first, then code, then gates.

## Tasks

| # | Task | Files | Tests |
|---|---|---|---|
| 1 | Shared files: pydantic floor `>=2.12,<3`, every extra of §18.1, pytest marker, mypy overrides, coverage omit, entry-point header; `noxfile.py` (`core`, `core-floor`, `lint`, `typecheck`, `docs`, `coverage`, `EXTRAS`) | `pyproject.toml`, `noxfile.py`, `SHARED-EDITS.md` | `nox -s core-floor` on 2.12.x |
| 2 | Errors: `kind`/`retryable` on every class, all new classes of §3.1, pickling | `core/errors.py` | `tests/core/test_errors.py` (+ new cases) |
| 3 | Value types: `ToolCall`, `ErrorInfo.from_exception`, `ToolResult`, `Event`, `Usage`, `LedgerEntry`, approval types, `ModelCaps` | `core/types.py` | `tests/core/test_types.py` |
| 4 | Message types: `Message`, `ToolDef`, `OutputSchema`, `LLMRequest`, `LLMResponse.parse`, stream events | `core/messages.py` | `tests/core/test_messages.py` |
| 5 | Ports (runtime-checkable protocols) | `core/ports.py` | `tests/core/test_ports.py` |
| 6 | `ToolContext` new fields, `emit`, `child` | `core/context.py` | `tests/core/test_context.py` |
| 7 | `ToolSpec` fields, `json_schema`/`result_schema` overrides, `tool_def`, policy checks, extras forwarded | `core/tool.py`, `core/_policy.py` | `tests/core/test_toolspec.py`, `test_policy.py` |
| 8 | `build_spec(args_model=)` | `core/tool.py`, `core/_signature.py` | `tests/core/test_args_model.py` |
| 9 | `model_from_json_schema` | `schema/models.py` | `tests/schema/test_models.py` |
| 10 | `spec_from_schema` | `core/imported.py` | `tests/core/test_imported.py` |
| 11 | `Registry.add_many`, `for_target` | `core/registry.py` | `tests/core/test_registry.py` |
| 12 | Renderers honour `targets`; OpenAI strict downgrade for non-portable imported tools | `integrations/{openai,gemini,anthropic}.py` | `tests/integrations/test_renderers.py` |
| 13 | Runtime: `run_sync`, `LoopLocal`, `current_context`/`use_context`, `new_id` | `runtime/*` | `tests/runtime/*` |
| 14 | Hooks base + dispatch | `hooks/{base,dispatch}.py` | `tests/hooks/*` |
| 15 | Event catalogue: `EventType`, `REQUIRED_DATA`, `make_event`, `emit` | `tracing/events.py` | `tests/tracing/test_events.py` |
| 16 | Cache keys, codec, `ToolResultCache` | `cache/{keys,codec,tool_cache}.py` | `tests/cache/*` |
| 17 | Shared fakes | `tests/fakes/*` | used by tasks 6, 14–16; `tests/fakes/test_fakes.py` |
| 18 | Boundary test R1–R9 with detector self-tests | `tests/test_boundaries.py` | itself |
| 19 | Docs: new `concepts/context.md`, `concepts/runtime.md`; update tools, errors, registry, schemas, tool-lifecycle, schema-pipeline; features rows split per §22; nav; CHANGELOG; exports | `docs/**`, `examples/*`, `mkdocs.yml`, `CHANGELOG.md`, `agent_adapter/__init__.py` | `tests/test_examples.py`, `mkdocs build --strict` |

## Exit gate

- `pytest -q`: the 0.1.0 suite unchanged and green, 0 warnings, coverage ≥ 90 % on new code.
- `ruff check .`, `ruff format --check .`, `mypy` (strict) clean; `mkdocs build --strict`.
- `nox -s core core-floor lint typecheck docs` green (`core-floor` on pydantic 2.12.x).
- Boundary detectors self-tested for R5–R9; no file over 500 lines.
