# Schema pipeline

Each tool has one JSON schema, built by pydantic from its args model. Each provider gets
a rewritten copy. You never call the transforms yourself; the renderers do.

## The pipeline

```mermaid
flowchart TB
  A["args model<br/>(pydantic)"] --> J["spec.json_schema()"]
  IS["input_schema<br/>(imported tool)"] --> J
  J --> CP{"check_portable<br/>at registration"}
  CP -- problems --> X["SchemaNotPortable"]
  CP -- ok --> PL["to_plain<br/>inline $ref · flatten 1-item allOf<br/>drop title · const → enum"]
  PL --> ANT["Anthropic<br/>input_schema<br/>(defaults kept)"]
  PL --> NS["OpenAI strict=False<br/>(plain, defaults kept)"]
  PL --> DD["defaults → descriptions<br/>'Default: 8.'"]
  DD --> ST["OpenAI strict<br/>all required · optional nullable<br/>additionalProperties: false"]
  DD --> GM["Gemini<br/>accepted keywords only"]
```

| Function (`schema.transforms`) | Used by |
|---|---|
| `to_plain` | every renderer |
| `to_openai_parameters(schema, strict=True)` | `chat_tool(s)`, `responses_tool(s)` |
| `to_gemini_parameters` | `function_declaration(s)`, `genai_tool` |
| `to_anthropic_input_schema` | `anthropic_tool(s)` |

`json_schema()` returns `input_schema` unchanged when a tool was built by
`spec_from_schema`; otherwise the args model's schema. Every renderer first narrows the
registry with `for_target`. One rule turns strict off: an imported schema that is not
portable renders with `strict: false` on OpenAI and in `ToolSpec.tool_def()`.

## Strict mode, step by step

Strict mode makes every field required. An optional field becomes nullable. The model
then sends `null` to mean "not given". The executor runs the call, and `validate_args`
turns that `null` back into the default.

```mermaid
sequenceDiagram
  participant Py as Signature
  participant Sch as Schema
  participant LLM as OpenAI
  participant Inv as executor → validate_args
  Py->>Sch: top_k: int = 8
  Note over Sch: integer or null, required
  LLM->>Inv: top_k: null
  Inv->>Inv: drop the null
  Inv->>Py: top_k=8
```

| Field declared as | Model sends `null` | Function receives |
|---|---|---|
| `top_k: int = 8` | dropped before validation | `8` |
| `region: str | None = None` | kept | `None` |
| `query: str` (required) | kept | `ToolArgsInvalid` |

The rule applies at any depth: nested models, and models inside lists. Details on
[Strict mode and nulls](../concepts/strict-mode.md).

## Where defaults go

| Target | `default` keyword | Argument default in the description |
|---|---|---|
| OpenAI strict | removed | yes: `Default: 8.` |
| OpenAI `strict=False` | kept | no |
| Gemini | removed | yes: `Default: 8.` |
| Anthropic | kept | no |

Every property gets the note, at any depth — including fields of models inside lists.
A `default` placed directly on a list item type or a union branch (which pydantic
ignores) is dropped without a note.

## Why validation stays safe

Validation always uses the tool's own pydantic model. A keyword dropped for a provider
never weakens checking on our side.
