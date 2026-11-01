# Tools

A **tool** is a callable plus what an LLM needs to call it: a name, a description and an
argument schema. In code it is a `ToolSpec`. You never build one by hand.

## Two ways in, same result

| | `@registry.tool` | `registry.wrap(func)` |
|---|---|---|
| Use for | code you own | code you cannot or will not edit; bound methods |
| Options live | on the decorator | at the call site |
| Returns | the function, unchanged | the `ToolSpec` |

```python
--8<-- "examples/registry_views.py:tools"
```

`@registry.tool` also works without parentheses. `agent_adapter.tool` and
`agent_adapter.wrap` register into `agent_adapter.default_registry`.

See [Wrap existing code](../get-started/wrap-existing-code.md) for methods, partials and
callable objects.

## Where the schema comes from

| Source | Becomes |
|---|---|
| function name (or `name=`) | tool name: `[A-Za-z_][A-Za-z0-9_]*`, max 64 |
| docstring text before the first section (or `description=`) | tool description; required |
| Google-style `Args:` section | per-argument descriptions |
| `Annotated[T, Field(...)]` or `= Field(...)` | description, constraints, default |
| type hints | argument types |
| defaults | optional arguments |
| return annotation | `spec.returns`, `spec.result_schema()` |

A `Field(description=...)` beats the docstring for that argument.

## Rejected at build time

`build_spec` raises `ToolDefinitionError` for:

- `*args`, `**kwargs`, positional-only or unannotated parameters;
- `alias=` on a parameter (use `rename=`);
- no description and no docstring;
- more than one `ToolContext` parameter;
- an LLM-facing argument name that does not match `[A-Za-z_][A-Za-z0-9_]*` (max 64),
  starts with `_`, or clashes with a pydantic `BaseModel` attribute (`schema`, `json`,
  `copy`, `model_*`, ...). Hide or rename such parameters.

## Shaping options

| Option | Effect |
|---|---|
| `fixed={"lang": "en"}` | bound value, hidden from the LLM |
| `hide=["debug"]` or `hide="debug"` | hidden; the plain default applies |
| `rename={"article_list": "articles"}` | the LLM sees `articles`; the function gets `article_list` |
| `result=to_hits` | transforms the return value; `result_schema()` becomes `None` |
| `name=`, `description=` | override the function name and docstring |

`spec.llm_to_python` holds the reverse rename map: `{"articles": "article_list"}`.

## Policy options

Stored on the `ToolSpec` and type-checked at build time. A wrong type or an unknown
option raises `ToolDefinitionError`. The [Executor](executor.md) enforces the execution
options on every call; `spec.invoke` ignores them.

| Option | Type | Acted on by |
|---|---|---|
| `kind` | non-empty `str` (default `"tool"`) | `registry.subset(kinds=)`; executor `RunLimits.per_kind` caps |
| `tags` | one non-empty `str`, or an iterable of them | `registry.subset(tags=)` |
| `prompt_hint` | `str` or `None` | `registry.catalogue()` |
| `portable` | `bool` (default `True`) | registration |
| `parallel_safe`, `once_per_run` | `bool` | [executor](executor-policies.md#runs-and-guards): run lock, run guard |
| `confirm` | `bool` or `(call, ctx) -> bool` (sync or async) | [executor](executor-policies.md#approval) → approval port |
| `timeout`, `cache_ttl` | finite positive number (seconds) or `None` | [executor](executor-policies.md#retry-timeout-and-control-flow) and its tool cache |
| `retries` | `int >= 0` (extra attempts) | [executor](executor-policies.md#retry-timeout-and-control-flow) |
| `cache_scope` | `"run"` (default), `"thread"` or `"global"` | who shares a cached result ([executor](executor-policies.md#cache-ledger-and-events)) |
| `targets` | one non-empty `str`, or an iterable of them; default `None` = every target | renderers and exporters ([Registry](registry.md#targets)) |
| `effects` | `"read_only"`, `"additive"`, `"destructive"` or `None` | MCP tool annotations |
| `idempotent` | `bool` or `None` | MCP tool annotations |

## Bring your own argument model

When the signature cannot describe the arguments well, pass `args_model=` a pydantic
model. Its field names must equal the LLM-facing parameter names (after `fixed`, `hide`,
`rename` and the context parameter are removed); it must use `extra="forbid"` and no
aliases. The model's types and descriptions are what validation and the schema use.
Parameter annotations are still required on the function.

```python
--8<-- "examples/imported_tools.py:args_model"
```

A mismatch raises `ToolDefinitionError` naming the missing and extra fields.

## Tools without a signature

`spec_from_schema(name, description, input_schema, func, ...)` builds a tool from a raw
JSON schema, for tools imported from MCP, LangChain or OpenAPI. See
[Schemas](schemas.md#tools-from-a-raw-json-schema).

## Context

A parameter annotated `ToolContext` is injected at call time and never appears in the
schema. A subclass, or `ToolContext | None`, works too. Only one is allowed. If you pass
no `ctx`, the tool gets an empty `ToolContext()`. Fields, `emit` and `child` are on
[Context](context.md).

## Calling a tool

In an application, and for every tool call a model sends, call tools through the
[Executor](executor.md) so approval, budgets, guards and the other policy options apply.
The spec methods run the function alone: use them only in unit tests.

```python
--8<-- "examples/registry_views.py:call"
```

| Method | Does | Raises |
|---|---|---|
| `validate_args(args)` | strict nulls → defaults, then validate | `ToolArgsInvalid` |
| `call(validated, ctx)` | run the function; sync ones in a worker thread | `ToolUsageError` for another tool's model |
| `invoke(args, ctx)` | both | both |
| `json_schema()` | provider-neutral argument schema (a fresh copy; `input_schema` when set) | — |
| `result_schema()` | `output_schema`, else the return annotation's schema, or `None` | — |
| `tool_def(strict=True)` | the tool as a `ToolDef` for model ports | — |

Unknown arguments are rejected. `ToolSpec` compares and hashes by identity.
