# HITL, cost and cache

How approval, the cost ledger, the caches and the model ports fit together around one
agent step. Each layer has a concept page with the details; this page shows the wiring.

```mermaid
flowchart LR
  subgraph step["one agent step"]
    M["InstrumentedModel<br/>(executor.model)"] --> X["Executor<br/>run_many"]
  end
  M -- "llm entry" --> L[("Ledger")]
  X -- "tool entry" --> L
  M -- "llm.*" --> E[("EventSink")]
  X -- "tool.* · approval.*" --> E
  M <--> RC[("LLM response cache")]
  X <--> TC[("tool-result cache")]
  X <--> A["ApprovalPort"]
  RC --- C[("CachePort<br/>Memory · Redis")]
  TC --- C
```

## Approval (HITL)

A tool with `confirm=True` (or a rule) goes through the executor's `ApprovalPort` after the
`before_tool` hooks; the decision may edit the arguments, which are validated again.
Details: [Approval](../concepts/approval.md).

```mermaid
sequenceDiagram
  participant Ex as Executor
  participant Port as ApprovalPort
  participant Hu as Human
  participant T as Tool
  Ex->>Ex: confirm rule says yes
  Ex->>Port: request(ApprovalRequest)
  alt approved
    Port-->>Ex: decision (edited_args?)
    Ex->>T: call
  else denied
    Port-->>Ex: approved=False → ApprovalDenied
  else deferred
    Port-->>Ex: ApprovalPending → result status "pending"
    Hu->>Port: store.decide(approval_id, decision)
    Ex->>Port: same run + call id again → decision, consumed once
  end
```

| Outcome | Executor | Tool loop |
|---|---|---|
| approved | runs with `edited_args` if given | the result goes back to the model |
| denied | `ApprovalDenied` (an error result in batches) | the error goes back to the model |
| pending | `ApprovalPending` / `status="pending"` | stops with `finish="pending"`; resume with the same `ctx` |

Approval records are bound to the tool and a digest of the original arguments, so an
approval never covers other arguments. Framework bridges (LangGraph, ADK, Agents SDK,
Pydantic AI, …) only add a gate: a call is approved by the executor's port or by a
one-use token the adapter issued after that port approved
([one-use rule](../how-to/human-in-the-loop.md#the-one-use-rule)). One opt-in exception: an MCP
server built with `client_approval=True` makes the MCP client's user the approver
([MCP approval](../integrations/mcp.md#approval)).

## Cost ledger

```mermaid
flowchart TB
  LLM["InstrumentedModel call"] --> U["usage normaliser<br/>OpenAI · Gemini · Anthropic"]
  U --> E["LedgerEntry kind=llm"]
  TOOL["executor tool call"] --> E2["LedgerEntry kind=tool"]
  API["paid API (Gemini cache storage)"] --> E3["LedgerEntry kind=api"]
  E --> LG["Ledger.record<br/>prices with Prices"]
  E2 --> LG
  E3 --> LG
  LG --> S["sinks: memory · callback · JSONL"]
  LG --> SUM["summary(run_id)"]
```

| Who records | `kind` | Entry id | Cost |
|---|---|---|---|
| `InstrumentedModel` | `llm` | `{call or run id}:llm:{stage}:{n}` or `metadata["call_id"]` | tokens × the price row; `0` on a response-cache hit |
| Executor | `tool` | the tool call id | `0` |
| `GeminiExplicitCache` | `api` | `gcache:{name}` | `cache_storage_per_hour` × token-hours |

Entries are idempotent on `(run_id, entry_id)`: a re-run with the same ids replaces them.
`BudgetHook` reads `ledger.total_usd(...)` before each tool and model call. Details:
[Cost](../concepts/cost.md), [Instrumented models](../concepts/tool-loop.md).

## Cache layers

```mermaid
flowchart TB
  REQ["model or tool call"] --> TC{"tool-result cache<br/>tool + args + cache_scope"}
  REQ --> LC{"LLM response cache<br/>provider + model + request + port fields"}
  LC -- miss --> PL["PromptLayout<br/>stable prefix first"]
  PL --> PC["prompt-cache plug-in<br/>caps from Prices.default()"]
  PC --> API["provider API"]
  TC -- hit --> OUT["result, cost 0"]
  LC -- hit --> OUT
```

| Layer | Where | Saves |
|---|---|---|
| Tool-result cache | executor, tools with `cache_ttl` | repeated retrievals, paid reranks |
| LLM response cache | `InstrumentedModel` with `cache=` (`executor.model`) | regenerations, eval re-runs: a second identical loop costs 0 |
| Prompt layout | `PromptLayout(static=..., dynamic=...)` | makes the provider's prefix cache hit |
| Provider plug-ins | `cache_plugin=` on a port | `prompt_cache_key` / `prompt_cache_options` (OpenAI), `cachedContent` (Gemini), `cache_control` (Anthropic) |

Details: [Cache](../concepts/cache.md).

## Model ports and the tool loop

Every port implements `LLMPort` (`generate`, `stream`, `cache_key_fields`): `OpenAIModel`,
`GeminiModel`, `AnthropicModel`, `CallableModel(fn)` for a host function, and
`FallbackModel([a, b])` for ordered fallback on transient errors. `InstrumentedModel` adds
the layers above to any of them. `run_tool_loop(model, registry, messages)` is the minimal
loop for apps without a framework; tool calls run through `executor.run_many`. There is no
model gateway (no LiteLLM). Details: [Model ports](../concepts/models.md),
[Instrumented models and the tool loop](../concepts/tool-loop.md).
