# Pydantic AI

:material-check-circle:{ .status-ok } Available · `pip install "agent-adapter[pydantic-ai]"` · `pydantic-ai-slim>=2.60`

Every registry tool becomes a Pydantic AI `Tool` (built with `Tool.from_schema` from the
tool's neutral JSON schema) that runs through the [executor](../concepts/executor.md). A
tool's `confirm=` rule drives Pydantic AI's deferred-approval flow, and approvals may edit
arguments natively.

## Functions

| Function (`agent_adapter.integrations.pydantic_ai`) | Returns |
|---|---|
| `pydantic_ai_tool(spec, *, executor=None)` | one `Tool` (`takes_ctx=True`) |
| `pydantic_ai_tools(registry, *, executor=None)` | one per tool whose `targets` include `pydantic_ai` |
| `async resolve_deferred(requests, port, *, run_id="")` | `DeferredToolResults` for `requests.approvals` |
| `INTEGRATION` | the [integration](../contributing/writing-an-integration.md) object; `bind_context(run_context)` reads its ids |

`executor=None` uses the default executor. Pydantic AI skips schema validation for
`from_schema` tools; the executor validates every call.

## Run and approve

```python
--8<-- "examples/pydantic_ai_tools.py:run"
```

```mermaid
sequenceDiagram
  participant Agent as agent.run
  participant T as Tool
  participant Host as Your app
  participant Port as ApprovalPort
  participant Ex as Executor
  Agent->>T: call (confirm rule says yes)
  T->>T: remember this call's ToolCallPart object
  T-->>Agent: ApprovalRequired (after validating)
  Agent-->>Host: output = DeferredToolRequests
  Host->>Port: resolve_deferred(requests, port)
  Port-->>Host: decision
  Note over Host: approved: ToolApproved(override_args) + a one-use token
  Host->>Agent: agent.run(message_history=..., deferred_tool_results=...)
  Agent->>T: call with (edited) args; the history holds the same ToolCallPart
  T->>T: take the token for exactly this call
  T->>Ex: run_spec(spec, args, ctx.approval = token's decision or None)
```

## Approval

The agent needs `output_type=[..., DeferredToolRequests]`, or Pydantic AI raises when a
tool asks for approval.

| `confirm=` | Pydantic AI gate |
|---|---|
| `False` | none |
| `True` | the tool validates the arguments, then raises `ApprovalRequired(metadata={"tool": name})` |
| callable `(call, ctx) -> bool` | the tool validates the arguments, asks the rule and raises `ApprovalRequired` for the calls it selects |

Invalid arguments are never sent for approval: they raise `ModelRetry`. A rule that says
"no" lets the call run past the Pydantic AI gate; the executor still runs the rule itself
before the tool runs.

**Pydantic AI's own approval never counts alone.** `tool_call_approved=True`, from any
`DeferredToolResults` (`approvals={id: True}`, a `ToolApproved` you built), only lets the
call through the gate. The executor skips its own port only for a token
`resolve_deferred` issued after **your** port approved.

| Token is bound to | So that |
|---|---|
| the `ToolCallPart` **object** the tool paused on (found again in the resumed run's `messages`) | another run, even with the same call, tool and arguments, gets no approval; nothing in `DeferredToolResults` can be copied to forge one |
| the `tool_call_id` and the tool | an approval covers no other call |
| a digest of the validated arguments the tool runs with (the edit, if any) | arguments changed after approval get no approval |
| one use, within 60 seconds | replaying the same results gets no approval |

Without a matching token the executor decides: its rule, then its own port (none: denied).

**What the binding is, exactly.** Pydantic AI gives a resumed run a new `run_id`, so the
token cannot name the run. It names the paused call's `ToolCallPart` object instead. Only a
resume whose `message_history` holds that same object (the paused run's own
`all_messages()`) can use it, once. Code in your process that passes run A's message
objects as another run's history is resuming run A.

- Tokens are issued only for calls this bridge's tools paused on; other tools get
  `ToolApproved` alone. One resolve issues at most 256 tokens; the rest, and any older
  token a full table drops, fall back to the executor's port (logged as a WARNING with
  the count).
- `resolve_deferred` asks the port once per call in `requests.approvals`. For this
  bridge's tools the request carries the **validated** arguments; for other tools the
  model's raw ones. It also carries `metadata={"integration": "pydantic_ai", "native":
  <the call's deferred metadata>}` and the `run_id` you pass. Use the paused run's
  `run_id`, the same on every pass.
- Approved → `ToolApproved(override_args=decision.edited_args)`; Pydantic AI replaces the
  arguments before the tool runs, and the executor validates them.
- Denied → `ToolDenied` whose message is a model-visible `approval-denied` error.
- Resume once, in the same process, within a minute, with the paused run's own message
  objects. A history you store and load again (`ModelMessagesTypeAdapter`) holds new
  objects: the executor's own port decides when the call runs.
- Calls deferred for external execution (`requests.calls`) are left to you.
- `DeferredApproval` (out-of-band review) is asked in two phases; see
  [Approval batches](../concepts/approval-batches.md): resolve again after each decision,
  with the same `requests` and `run_id`.

## Ids

| Context field | From `RunContext` |
|---|---|
| `ctx.call_id` | `tool_call_id` |
| `ctx.run_id` | `run_id` (a resumed run has a new one) |
| `ctx.thread_id` | `conversation_id` (kept across a deferred resume) |

## Errors to expect

| Case | What the model sees / what you get |
|---|---|
| `ToolArgsInvalid` | `ModelRetry`: the model is asked to fix its arguments |
| `ToolError`, `ApprovalDenied`, `GuardrailBlocked`, `RunLimitExceeded` | the tool returns `{"error": {"kind", "message"}}` |
| any other error (`ToolTimeout`, `ToolFailed`, ...) | raised from `agent.run` |
| `asyncio.CancelledError` | propagates |

Set `PYDANTIC_AI_NO_BANNER=1` in tests to silence Pydantic AI's start-up banner.
