# WP13a — Native-approval frameworks (plan)

**Scope:** contract §14.4 rows `agent_framework`, `agno`, `strands`, `haystack`, `claude_agent`,
`semantic_kernel`. Internals per R5 §A.1 and the probes in `.superpowers/research/probes/p_*.py`.

## Design

Every exported tool calls `runner_for(executor).run_spec(spec, args, ctx, call_id=…)` with a
context from `INTEGRATION.bind_context(native)`. A framework's own gate approving a call sets
`ctx.approval`, so the executor does not ask again (spec §9).

| Target | Native tool | Gate | How the tool learns "approved" | Call id |
|---|---|---|---|---|
| Agent Framework | `FunctionTool` subclass, `input_model=<schema>` | `approval_mode="always_require"` | invoked with a `FunctionInvocationContext` (only MAF's function loop passes one) | `tool_call_id` |
| Agno | `Function(parameters=…, skip_entrypoint_processing=True)` | `requires_confirmation` | `fc.function._run_context` set (inside an agent run) | `fc.call_id`; run id from the run context |
| Strands | `AgentTool` subclass | `strands_approval_hook(port, registry)` on `BeforeToolCallEvent` | grant in `invocation_state`, bound to `toolUseId` + input | `toolUseId` |
| Haystack | `Tool(function=, async_function=)` + injected `State` | `confirmation_hook(port, registry)` wrapping `ConfirmationHook` | grant in `state.data`, bound to tool call id + arguments | matched from the last assistant message |
| Claude Agent SDK | `SdkMcpTool` in `create_sdk_mcp_server` | `can_use_tool(port, registry)` | one-shot grant (tool + final input, TTL) consumed by the handler | the grant's `tool_use_id` |
| Semantic Kernel | `KernelFunctionFromMethod` with explicit metadata | `approval_filter(port, registry)` (auto-invocation filter) | context variable set by the filter around `next(context)` | `function_call_content.id` |

Shared helper (non-ODM, core coverage): `integrations/_native_gate.py` — validate, evaluate the
confirm rule, ask the port, refuse edits where unsupported (deny, fail closed), and the grant
table used by the Claude bridge.

## Tasks (test first, one commit each)

1. `_native_gate.py` + `tests/integrations/test_native_gate.py`.
2. `agent_framework.py` + `test_agent_framework_contract.py` (C1–C8; C6 through `Agent` with a
   fake `BaseChatClient` and `function_approval_response`) + unit tests in the same file.
3. `agno.py` + contract (C6 through `Agent.arun` / `acontinue_run` with a fake `Model`).
4. `strands.py` + contract (C6 through `Agent.invoke_async` with a fake `Model` + the hook).
5. `haystack.py` + contract (C2–C7 and C6 through `Agent.run_async` with a fake chat generator).
6. `claude_agent.py` + contract (kit calls the SDK tool handler as its in-process server does;
   extra tests go through `create_sdk_mcp_server` + `mcp.Client`; C6 through `can_use_tool`;
   tool-name mapping `mcp__<server>__<tool>`).
7. `semantic_kernel.py` + contract (calls through `Kernel.invoke_function_call`; C6 through the
   auto-invocation filter; tool-name mapping `<plugin>-<function>`); own env (pydantic < 2.14).
8. Docs: one `docs/integrations/<name>.md` each, `examples/<name>_tools.py` snippets (each
   run by `test_the_example_runs` in its contract file, so it runs in that extra's session);
   SHARED-EDITS (features rows, nav, integrations index, CHANGELOG, noxfile note).

## Exit gate

- Each contract file passes in its own env with ≥ 90 % line + branch coverage of its module
  (`nox -s contract-<name> -p 3.12`, envdir `D:\aa\nox-wp13a`, deleted afterwards).
- Core: `pytest -q` (0 warnings), `ruff check .`, `ruff format --check .`, `mypy`,
  `mkdocs build --strict`; boundary test green; no file over 500 lines.
