# Tool lifecycle

A tool goes through two phases. At **startup** a function becomes a checked `ToolSpec`.
At **request time** the spec is rendered for a model, and the model's call is validated
and run.

## Startup: registration

```mermaid
sequenceDiagram
  participant App as Your code
  participant Reg as Registry
  participant Spec as ToolSpec
  App->>Reg: @tool / wrap(func)
  Reg->>Spec: build_spec(func) or spec_from_schema(schema, func)
  Spec-->>Reg: spec
  Reg->>Reg: name free?
  Reg->>Reg: check_portable
  Reg-->>App: spec stored
```

## Request time: a tool call

```mermaid
sequenceDiagram
  participant App as Your code
  participant Ren as Renderer
  participant LLM as Model API
  participant Ex as Executor
  participant Spec as ToolSpec
  App->>Ren: render
  Ren-->>App: tools
  App->>LLM: messages, tools
  LLM-->>App: tool call
  App->>Ex: run_call(ToolCall(name, args, id))
  Ex->>Ex: hooks, approval, guards
  Ex->>Spec: invoke(args, ctx)
  Spec->>Spec: validate_args
  Spec->>Spec: call
  Spec-->>Ex: value
  Ex-->>App: ToolResult
  App->>LLM: tool result
```

| Step | What happens |
|---|---|
| `build_spec` | reads signature, docstring and options; builds the args model (or checks a given `args_model=`) |
| `spec_from_schema` | for imported tools: keeps the raw schema as `input_schema`, builds a loose args model |
| name free? | `DuplicateTool` if the name is taken |
| `check_portable` | `SchemaNotPortable` if a provider would reject the schema |
| render | `chat_tools`, `function_declarations` or `anthropic_tools` over `registry.for_target(...)`; uses a cached copy of `json_schema()` |
| `run_call` | `executor.run_call(ToolCall(name=..., arguments=..., id=...))`: hooks, approval, budgets and guards, then `invoke`; failures come back in `result.error` |
| `invoke` | the raw tool body, which the executor calls as `spec.invoke(...)`; never call it yourself for a model's tool call (it skips every policy) |
| `validate_args` | strict nulls → defaults, then pydantic validation |
| `call` | maps renamed args back, adds `fixed`, injects `ToolContext`, applies `result=` |

## Inside `build_spec`

```mermaid
flowchart TD
  subgraph BS["build_spec(func, **options)"]
    F["callable<br/>function · bound method · partial · callable object"] --> N["name<br/>name= or __name__"]
    N --> P["policy options<br/>type-checked"]
    P --> S["inspect.signature<br/>reject *args, **kwargs, positional-only"]
    S --> D["Google docstring<br/>summary + Args:"]
    D --> O["shaping<br/>fixed · hide · rename · partial keywords"]
    O --> FL["fields<br/>ToolContext param hidden; names checked"]
    FL --> DC{"description?<br/>description= or docstring"}
    DC -- yes --> M["args model<br/>pydantic, extra='forbid'"]
    M --> T["ToolSpec"]
  end
  subgraph RA["Registry.add(spec)"]
    DUP{"name taken?"} -- no --> C{"portable?<br/>(unless portable=False)"}
    C -- yes --> R["stored in Registry"]
  end
  T --> DUP
  DUP -- yes --> E1["DuplicateTool"]
  C -- no --> E["SchemaNotPortable"]
  DC -- no --> X["ToolDefinitionError"]
  S -. "bad signature" .-> X
```

Every check runs once, at registration. A bad tool stops the app at startup.

## Request time: `invoke`

| Step | Method | On failure |
|---|---|---|
| Copy args, drop strict-mode nulls | `validate_args` | — |
| Validate against the args model | `validate_args` | `ToolArgsInvalid` |
| Check the model belongs to this tool | `call` | `ToolUsageError` |
| Map LLM names back (`llm_to_python`), add `fixed`, inject `ToolContext` | `call` | — |
| Imported tools instead: set values by property name, no rejected nulls | `call` | — |
| Await async functions; run sync ones in `asyncio.to_thread` | `call` | the tool's own exception |
| Apply `result=` | `call` | the transform's own exception |

`invoke(args, ctx)` is `call(validate_args(args), ctx)`. Use the two halves to validate
early and run later.

## In an application

`spec.invoke(...)` is the raw tool body: argument validation and the call, nothing else.
The executor wraps it with hooks, approval, budgets and run guards, cache, timeout, retry
and the cost ledger, and validates edited arguments again. Call `invoke` directly only in
unit tests. See [Executor and flows](executor-and-flows.md).
