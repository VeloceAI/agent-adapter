# WP07 Integration kit — TDD plan

Contract: `docs/superpowers/specs/2026-10-29-interface-contract.md` rev 2 (§2.3 R8, §4.6,
§14.1–§14.3, §18.2 `plugin`, §19, §21.3 WP07). Research: R3 §7 (entry points, fixture wheel).
Branch `wp/07-integration-kit` from `main` 9f65e0b. Consumes WP01 only: never imports
`agent_adapter.exec` (WP02 owns `runner_for`) or any other Wave A module.

## Tasks

| # | Task | Files | Tests |
|---|---|---|---|
| 1 | Protocols `ToolRunner`, `Integration`, `FlowCompiler`; `IntegrationBase` (`export_tools` honours `targets`, default `bind_context` / `bind_approval`, `require`); `model_visible_error`; `MODEL_VISIBLE_ERRORS`; `check_requires` (private `_check_requires` since fix round 1) | `integrations/base.py` | `tests/integrations/test_base.py` |
| 2 | Built-in `INTEGRATION`s for the renderers (native call id → `ctx.call_id`); `tool_result_blocks` | `integrations/{openai,gemini,anthropic}.py` | `tests/integrations/test_builtin_integrations.py` |
| 3 | Discovery: `GROUP`, `available_integrations(refresh=)`, `get_integration`; warn + skip load failures, missing modules, failing constructors, non-conforming objects, name mismatches, duplicates; cache | `integrations/discovery.py` | `tests/integrations/test_discovery.py` (synthetic entry points; real ones when the demo plugin is installed) |
| 4 | Installable demo plugin: a reference third-party integration plus broken / missing / non-conforming entry points | `tests/fixtures/demo_integration/` | task 3 (plugin session), task 5 |
| 5 | Contract kit: `Harness`, `contract_tests` (C1–C8), `FakeRunner`; self-test against the demo integration, plus negative self-tests (a faulty harness fails each check) | `tests/integrations/kit.py`, `tests/integrations/test_kit_selftest.py` | itself |
| 6 | Docs: `contributing/writing-an-integration.md` (snippets from the demo plugin and self-test); Anthropic page gains `tool_result_blocks`; example + its test | `docs/contributing/writing-an-integration.md`, `docs/integrations/anthropic.md`, `examples/anthropic_results.py`, `tests/integrations/test_integration_examples.py` | example test, `mkdocs build --strict` |
| 7 | Shared edits: nav, features row, integrations index section, CHANGELOG, `plugin` nox session installs the build backend | `SHARED-EDITS.md` | `nox -s plugin` with the edit applied locally |

## Design notes

- The kit is typed against `ToolRunner` only. `contract_tests(factory, *, runner_factory=None)`:
  the default is the kit's `FakeRunner` (validation, confirm rule, approval port or
  `ctx.approval`, edited args, before/after hooks, timeout). The MA gate passes
  `runner_factory=Executor` (its constructor takes `hooks=` and `approval=`).
- `Harness` gains `describe(native)` and `expected_schema(spec)`: C1 cannot compare the
  exported schema without them.
- `Integration.name/requires/extra` are read-only properties in the protocol so that
  `IntegrationBase`'s `ClassVar`s satisfy it under mypy.

Historical note: this is the plan as first written. Fix rounds 1–2 (review) added
`_run_bound`, `_error_payload`, `_kit_support.py`, `demo_support.py`, `test_kit_faults.py`
and the optional harness members; see the WP07 report.

## Exit gate

- `pytest -q` green, 0 warnings, ≥ 90 % line + branch coverage on `integrations/base.py`,
  `integrations/discovery.py` and the renderer modules.
- `ruff check .`, `ruff format --check .`, `mypy` strict, `mkdocs build --strict`.
- `nox -s plugin` finds the demo plugin and skips the broken ones with warnings.
- Kit self-test runs C1–C8 against `FakeRunner`; the real-`Executor` run is the MA gate.
