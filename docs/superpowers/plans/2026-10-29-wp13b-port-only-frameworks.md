# WP13b Port-only frameworks (DSPy, smolagents, AutoGen) — TDD plan

Contract: `docs/superpowers/specs/2026-10-29-interface-contract.md` rev 3 (§7, §8, §14.1–§14.4
rows `dspy`, `smolagents`, `autogen`, §18, §19, §21.3 WP13b). Research: R5 §A.0–§A.1, probes
`p_dspy.py`, `p_smol.py`, `p_autogen.py`. Branch `wp/13b-port-only-frameworks` from `main`
42ee50a (WP01–WP07).

None of the three frameworks has a per-tool approval gate, so `bind_approval` returns `None`
and the executor's `ApprovalPort` decides (kit `approval = "port"`).

## Tasks

| # | Task | Files | Tests |
|---|---|---|---|
| 1 | DSPy: `dspy_tool(spec, *, executor=None) -> dspy.Tool` (`args` = plain-schema properties, `arg_types` from JSON types, async `acall` path, sync `__call__` via `run_sync`), `dspy_tools(registry, …)`, `INTEGRATION` | `integrations/dspy.py` | `tests/integrations/test_dspy_contract.py` (kit C1–C8 on `FakeRunner` and `Executor`; ReAct offline with `DummyLM`; missing-extra path) |
| 2 | smolagents: `smolagents_tool(spec, *, executor=None) -> Tool` (subclass factory, `inputs` in AUTHORIZED_TYPES with `nullable`, `output_type` from the result schema, sync `forward` via `run_sync`), `smolagents_tools`, `INTEGRATION` | `integrations/smolagents.py` | `tests/integrations/test_smolagents_contract.py` (kit with `cancels_tool=False`; `ToolCallingAgent.run()` with a fake model; sync bridge inside a running loop) |
| 3 | AutoGen: `autogen_tool(spec, *, executor=None) -> BaseTool` (`BaseTool[args_model, Any]` subclass, raw args validated by the runner, native call id from `run_json(call_id=)`, `CancellationToken` linked), `autogen_tools`, `INTEGRATION` | `integrations/autogen.py` | `tests/integrations/test_autogen_contract.py` (kit through `StaticWorkbench`; `AssistantAgent` with a fake chat client) |
| 4 | Docs + examples: one page each, examples run in the contract files | `docs/integrations/{dspy,smolagents,autogen}.md`, `examples/{dspy_tools,smolagents_tools,autogen_tools}.py` | example tests in each contract file, `mkdocs build --strict` (nav applied locally) |
| 5 | Shared edits: nav, features rows, integrations index rows, CHANGELOG, nox session install for the AutoGen test client if needed | `SHARED-EDITS.md` | `nox -s contract-dspy contract-smolagents contract-autogen` |

## Design notes

- Every exported tool runs `runner_for(executor).run_spec(spec, args, ctx, call_id=…)` at
  call time (so `set_default_executor` applies later) with a context from `bind_context`.
- DSPy and smolagents have no tool-error channel: `ToolArgsInvalid`, `ToolError`,
  `ApprovalDenied`, `GuardrailBlocked`, `RunLimitExceeded` come back as
  `model_visible_error(exc)` (JSON text for smolagents string tools); everything else, and
  control flow, propagates. AutoGen's channel is an exception from `run_json`, which its
  workbench reports as `ToolResult(is_error=True)`, so adapter errors raise there.
- Arguments are validated by our `args_model` in the runner, never by the framework, so
  invalid args never surface as a raw `pydantic.ValidationError`.
- Edits: the executor's port applies `edited_args` for every port-mode integration, so the
  kit's edit case runs (contract deviation from the §14.4 "no", recorded in the report).

## Exit gate

- `pytest -q` green, 0 warnings; `ruff check .`, `ruff format --check .`, `mypy` strict,
  `mkdocs build --strict` (shared nav applied locally only).
- `nox -s contract-dspy`, `contract-smolagents`, `contract-autogen` each alone with
  `--cov-fail-under=90` on its module; envdirs on D: deleted afterwards.
