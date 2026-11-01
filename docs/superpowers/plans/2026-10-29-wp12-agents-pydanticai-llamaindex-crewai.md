# WP12 Agents SDK, Pydantic AI, LlamaIndex, CrewAI — TDD plan

Contract: `docs/superpowers/specs/2026-10-29-interface-contract.md` rev 3 (§7, §8, §14.1–§14.4,
§18, §19, §21.3 WP12). Research: R3 §2–§5, §9. Branch `wp/12-agents-pydanticai-llamaindex-crewai`
from `main` 42ee50a (WP01–WP07). Every exported tool runs
`runner_for(executor).run_spec(spec, args, ctx, call_id=<native call id>)`.

## Tasks

| # | Task | Files | Tests |
|---|---|---|---|
| 1 | OpenAI Agents SDK: `function_tool` (strict params via `to_openai_parameters`, `needs_approval` from `confirm`, failure output for model-visible errors), `function_tools`, `ApprovalOverrides`, `resolve_interruptions` (result or `RunState`, port decisions → `approve` / `reject`, edits into the overrides table), `INTEGRATION` | `integrations/openai_agents.py` | `tests/integrations/test_openai_agents_contract.py` (kit through `Runner.run` with a fake `Model`), `tests/integrations/test_openai_agents.py` |
| 2 | Pydantic AI: `pydantic_ai_tool` (`Tool.from_schema`, `requires_approval` for `confirm=True`, `ApprovalRequired` for callable rules, `ModelRetry` for `ToolArgsInvalid`), `pydantic_ai_tools`, `resolve_deferred` (`ToolApproved(override_args)` / `ToolDenied`), `INTEGRATION` | `integrations/pydantic_ai.py` | `tests/integrations/test_pydantic_ai_contract.py` (kit through `Agent.run` with `FunctionModel`), `tests/integrations/test_pydantic_ai.py` |
| 3 | LlamaIndex: `llamaindex_tool` (`FunctionTool.from_defaults(fn=<run_sync bridge>, async_fn=…, fn_schema=spec.args_model)`), `llamaindex_tools`, `INTEGRATION` | `integrations/llamaindex.py` | `tests/integrations/test_llamaindex_contract.py` (`acall`), `tests/integrations/test_llamaindex.py` (sync `call` in and out of a loop) |
| 4 | CrewAI: `crewai_tool` (`BaseTool` subclass per spec, `_run` via `run_sync`, `_arun` async, result cache off), `crewai_tools`, `INTEGRATION`; own venv only | `integrations/crewai.py` | `tests/integrations/test_crewai_contract.py` (`arun`), `tests/integrations/test_crewai.py` (`run` inside a running loop) |
| 5 | Docs + examples: four pages flipped to Available; examples run offline with fake models | `docs/integrations/{openai-agents,pydantic-ai,llamaindex,crewai}.md`, `examples/{openai_agents,pydantic_ai,llamaindex,crewai}_tools.py`, `tests/test_examples_wp12.py` | example tests, `mkdocs build --strict` |
| 6 | Shared edits: features rows, CHANGELOG, integrations index | `SHARED-EDITS.md` | — |

Each task: write the contract test (kit) and unit tests first, watch them fail on the missing
module, implement, then run the module's own nox session.

## Design notes

- Native call ids: Agents SDK `ToolContext.tool_call_id`, Pydantic AI `RunContext.tool_call_id`;
  LlamaIndex and CrewAI have none at the tool boundary (`native_call_ids = False`).
- Native approval (Agents SDK, Pydantic AI): after the framework's gate approved a call the
  bridge sets `ctx.approval`, so the executor does not ask its port again. Edits: Agents SDK via
  the `ApprovalOverrides` side table keyed by `tool_call_id`; Pydantic AI natively
  (`override_args` replace the arguments before the tool runs).
- Port approval (LlamaIndex, CrewAI): the executor's approval port decides; no edits.
- Errors: Agents SDK and LlamaIndex/CrewAI return `model_visible_error` for the §14.1 set;
  Pydantic AI raises `ModelRetry` for `ToolArgsInvalid`. Everything else and control flow propagate.
- Heavy imports (LlamaIndex, CrewAI) are lazy (inside functions / `TYPE_CHECKING`).

## Exit gate

- `pytest -q` green with 0 warnings in the main venv (openai-agents, pydantic-ai, llamaindex
  installed); `ruff check .`, `ruff format --check .`, `mypy` strict, `mkdocs build --strict`.
- `nox -s contract-openai_agents-3.12 contract-pydantic_ai-3.12 contract-llamaindex-3.12
  contract-crewai-3.12 --envdir D:\aa\nox-wp12`: kit C1–C8 and ≥ 90 % coverage per module.
- Agents SDK approve / reject / edit through `ApprovalOverrides` with a fake `Model`;
  Pydantic AI static + dynamic approval, edit and deny with `FunctionModel`; CrewAI 1.15 inside
  a running loop.
