# Schemas

Each tool has one JSON schema, built by pydantic. At registration it is checked against
a **portable subset**: the features every supported provider accepts. Each renderer then
rewrites a copy for its provider. The full flow is on
[Schema pipeline](../architecture/schema-pipeline.md).

## The portable subset

| Allowed | Rejected |
|---|---|
| `str`, `int`, `float`, `bool` | `dict[str, T]` (open maps) |
| `Literal` / `Enum` of strings | non-string enum values |
| `list[T]` | tuples; a list with no item type |
| nested `BaseModel` | recursive models |
| `T | None` (also `type: ["X", "null"]`) | unions of several types (`int | str`), `oneOf`, discriminated unions |
| `Field(description=, ge=, le=)` | `pattern`, string `min_length`/`max_length`, `uniqueItems`, min/max properties |
| `format: date-time` | any other `format` |
| single-item `allOf` | `allOf` with several items |
| | `Any` (untyped value); a `type` list with no non-null type |

The root must be an object. A failure raises `SchemaNotPortable` at registration. Its
`problems` list names each path, e.g. `$.counts: open map (dict) is not portable`.

!!! tip "Need something outside the subset?"
    Pass `portable=False` to skip the check. Then test that tool on each provider you
    use: a provider may reject the schema at request time.

## What each provider gets

| Rewrite | OpenAI strict | OpenAI `strict=False` | Gemini | Anthropic |
|---|---|---|---|---|
| `$ref` inlined, `title` dropped, `const` → `enum` | yes | yes | yes | yes |
| every field required, optional made nullable | yes | no | no | no |
| `additionalProperties: false` on every object | yes | as given | as given | as given |
| `default` | removed; written into the description | kept | removed; written into the description | kept |
| keywords the provider does not accept | — | — | dropped | — |

Gemini keeps only: `type`, `description`, `properties`, `required`, `enum`, `format`,
`minimum`, `maximum`, `items`, `minItems`, `maxItems`, `anyOf`, `additionalProperties`,
`nullable`.

## One tool, three schemas

The `get_weather` tool below renders like this (output of `examples/providers.py`):

```python
--8<-- "examples/providers.py:define"
```

=== "OpenAI strict"

    ```json
    "unit": {
      "anyOf": [{"enum": ["c", "f"], "type": "string"}, {"type": "null"}],
      "description": "Temperature unit. Default: \"c\"."
    }
    ```
    `required` lists both `city` and `unit`.

=== "Gemini"

    ```json
    "unit": {
      "description": "Temperature unit. Default: \"c\".",
      "enum": ["c", "f"],
      "type": "string"
    }
    ```
    `required` lists only `city`.

=== "Anthropic"

    ```json
    "unit": {
      "default": "c",
      "description": "Temperature unit.",
      "enum": ["c", "f"],
      "type": "string"
    }
    ```
    `required` lists only `city`.

## Tools from a raw JSON schema

Tools imported from MCP, LangChain or OpenAPI arrive with a JSON schema, not a Python
signature. `spec_from_schema` keeps that schema as `input_schema` (and the result schema
as `output_schema`), so `json_schema()` returns it unchanged.

```python
--8<-- "examples/imported_tools.py:schema"
```

```python
--8<-- "examples/imported_tools.py:call"
```

```text
{'query': 'returns'}
{'query': 'returns', 'max-results': 2}
```

`func` gets the arguments the model actually set, named as the schema's properties
(`max-results` included). Unset optional arguments are not sent, so the remote side
applies its own defaults; a `null` is dropped unless the property schema allows null.

Arguments are validated by a **loose** model (`schema.models.model_from_json_schema`):

| Property schema | Field type |
|---|---|
| `string` with string `enum` | `Literal[...]` |
| `string` / `integer` / `number` / `boolean` | `str` / `int` / `float` / `bool` |
| `type: [X, "null"]`, or `anyOf` X and null, X primitive | `X \| None` |
| `array` / `object` | `list[Any]` / `dict[str, Any]` |
| anything else (`oneOf`, `allOf`, `$ref`, no type) | `Any` |

Required properties are required; unknown properties are rejected unless the root sets
`additionalProperties`. Names that are not valid field names (`max-results`, `_id`,
`schema`) get a sanitised field name with the original as alias; both validate.

| `portable=` | Effect |
|---|---|
| `None` (default) | `check_portable` decides; problems are logged at `DEBUG` |
| `True` | `registry.add` raises `SchemaNotPortable` if the schema is not portable |
| `False` | no check |

An imported tool whose schema is not portable renders with `strict: false` on OpenAI and
in `tool_def()`, whatever `strict` you ask for. That is the only strict downgrade.

## Validation is always ours

Incoming arguments are validated with the tool's own pydantic model, never by trusting
the provider. A keyword dropped for a provider never weakens checking.

## Return schemas

`spec.result_schema()` describes the return annotation. It is `None` when there is no
annotation, the annotation is `-> None`, a `result=` transform is set, or pydantic cannot
describe the type (a warning is logged). It never blocks registration.
