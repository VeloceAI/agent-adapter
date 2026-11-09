# Integrations

Each integration is one module under `agent_adapter.integrations` and, where it needs a
framework, one extra (install with `pip install "agent-adapter[<extra>]"`). Framework imports are
lazy, so an integration costs nothing until you call it. Each must pass a **contract
test** that runs a real tool through that framework and the real `Executor`.

<div class="aa-filter" role="group" aria-label="Filter integrations"><button type="button" data-kind="all" aria-pressed="true">All</button><button type="button" data-kind="framework" aria-pressed="false">Agent frameworks</button><button type="button" data-kind="api" aria-pressed="false">Model APIs</button><button type="button" data-kind="protocol" aria-pressed="false">Protocols and tools</button></div>

<div class="aa-gallery" markdown>
[<b>Composio</b><span>Hosted app tools (Gmail, Slack, GitHub...) through MCP</span><small>extra: mcp</small>](composio.md){ .aa-gcard .aa-gcard--hl data-kind="protocol" }
[<b>OpenAI</b><small>extra: none (plain dicts)</small>](openai.md){ .aa-gcard data-kind="api" }
[<b>Gemini</b><small>extra: gemini</small>](gemini.md){ .aa-gcard data-kind="api" }
[<b>Anthropic</b><small>extra: none (plain dicts)</small>](anthropic.md){ .aa-gcard data-kind="api" }
[<b>LangChain</b><small>extra: langchain</small>](langchain.md){ .aa-gcard data-kind="framework" }
[<b>OpenAPI</b><small>extra: openapi</small>](openapi.md){ .aa-gcard data-kind="protocol" }
[<b>MCP</b><small>extra: mcp</small>](mcp.md){ .aa-gcard data-kind="protocol" }
[<b>OpenAI Agents SDK</b><small>extra: openai-agents</small>](openai-agents.md){ .aa-gcard data-kind="framework" }
[<b>Pydantic AI</b><small>extra: pydantic-ai</small>](pydantic-ai.md){ .aa-gcard data-kind="framework" }
[<b>LlamaIndex</b><small>extra: llamaindex</small>](llamaindex.md){ .aa-gcard data-kind="framework" }
[<b>CrewAI</b><small>extra: crewai (own environment)</small>](crewai.md){ .aa-gcard data-kind="framework" }
[<b>Microsoft Agent Framework</b><small>extra: agent-framework</small>](agent-framework.md){ .aa-gcard data-kind="framework" }
[<b>Agno</b><small>extra: agno</small>](agno.md){ .aa-gcard data-kind="framework" }
[<b>Strands Agents</b><small>extra: strands (own environment)</small>](strands.md){ .aa-gcard data-kind="framework" }
[<b>Haystack</b><small>extra: haystack</small>](haystack.md){ .aa-gcard data-kind="framework" }
[<b>Claude Agent SDK</b><small>extra: claude-agent</small>](claude-agent.md){ .aa-gcard data-kind="framework" }
[<b>Semantic Kernel</b><small>extra: semantic-kernel (own environment)</small>](semantic-kernel.md){ .aa-gcard data-kind="framework" }
[<b>DSPy</b><small>extra: dspy</small>](dspy.md){ .aa-gcard data-kind="framework" }
[<b>smolagents</b><small>extra: smolagents</small>](smolagents.md){ .aa-gcard data-kind="framework" }
[<b>AutoGen 0.7</b><small>extra: autogen</small>](autogen.md){ .aa-gcard data-kind="framework" }
[<b>LangGraph</b><small>extra: langgraph</small>](langgraph.md){ .aa-gcard data-kind="framework" }
[<b>Google ADK 2.x</b><small>extra: adk (own environment)</small>](adk.md){ .aa-gcard data-kind="framework" }
[<b>CAMEL-AI</b><small>extra: camel (own environment)</small>](camel.md){ .aa-gcard data-kind="framework" }
[<b>AG2</b><small>extra: ag2</small>](ag2.md){ .aa-gcard data-kind="framework" }
[<b>A2A</b><small>extra: a2a</small>](a2a.md){ .aa-gcard data-kind="protocol" }
[<b>AG-UI</b><small>extra: ag-ui</small>](ag-ui.md){ .aa-gcard data-kind="protocol" }
</div>

Not on this list? [Connect anywhere](../connect/index.md) attaches 12 more stacks through MCP,
REST or the TS client.

## Matrix

| Target | Exports | Also | Status |
|---|---|---|---|
| [OpenAI](openai.md) (Chat / Responses) | tool dicts, strict schemas | — | :material-check-circle:{ .status-ok } Available |
| [Gemini](gemini.md) | function declarations, `genai_tool()`, AFC callables | `run_function_calls` | :material-check-circle:{ .status-ok } Available |
| [Anthropic](anthropic.md) | `input_schema` tools | — | :material-check-circle:{ .status-ok } Available |
| [Microsoft Agent Framework](agent-framework.md) | `FunctionTool`s | `approval_mode="always_require"`, answered by `resolve_approvals(response, port, registry)` | :material-check-circle:{ .status-ok } Available |
| [Agno](agno.md) | `Function`s | `requires_confirmation`, resolved by `resolve_requirements(output, port, registry)` | :material-check-circle:{ .status-ok } Available |
| [Strands Agents](strands.md) | `AgentTool`s | `strands_approval_hook(port, registry)` | :material-check-circle:{ .status-ok } Available |
| [Haystack](haystack.md) | `Tool`s | `confirmation_hook(port, registry)` | :material-check-circle:{ .status-ok } Available |
| [Claude Agent SDK](claude-agent.md) | SDK MCP server | `claude_sdk_bridge(registry, port)`: server + its own `can_use_tool` (edits via `updated_input`) | :material-check-circle:{ .status-ok } Available |
| [Semantic Kernel](semantic-kernel.md) | kernel functions, plugin | `approval_filter(port, registry)` | :material-check-circle:{ .status-ok } Available |
| [DSPy](dspy.md) | `dspy.Tool` per spec (`acall`; sync `__call__` via `run_sync`) | — | :material-check-circle:{ .status-ok } Available |
| [smolagents](smolagents.md) | `Tool` subclass per spec (sync `forward` via `run_sync`) | — | :material-check-circle:{ .status-ok } Available |
| [AutoGen 0.7](autogen.md) | `BaseTool` per spec (native call id, `CancellationToken`) | — | :material-check-circle:{ .status-ok } Available |
| [LangGraph](langgraph.md) | `tool_node` (interrupt approvals, one at a time), `tool_task` + `fan_out` (`Send`), `reducer(rule)` channels | `compile_flow(flow) -> StateGraph`, `run_compiled` (parity with the local runner) | :material-check-circle:{ .status-ok } Available |
| [Google ADK 2.x](adk.md) | `BaseTool` per spec; confirmations answered by `resolve_confirmations(events, port, registry)` (one-use tokens, edits) | `compile_flow(flow) -> Workflow` (superstep driver), `run_workflow`, `resume_message` | :material-check-circle:{ .status-ok } Available |
| [LangChain](langchain.md) | `StructuredTool`s | importer: `import_langchain` | :material-check-circle:{ .status-ok } Available |
| [MCP](mcp.md) | lowlevel `Server` and stateless Streamable HTTP app over `for_target("mcp")`; annotations from `effects`; principal-scoped ids | `import_mcp` importer, `mcp_client` | :material-check-circle:{ .status-ok } Available |
| [OpenAI Agents SDK](openai-agents.md) | `FunctionTool`s with `needs_approval` | `resolve_interruptions` (edits travel in one-use tokens) | :material-check-circle:{ .status-ok } Available |
| [OpenAPI](openapi.md) | — | importer: `import_openapi(registry, spec, base_url=, auth=)` | :material-check-circle:{ .status-ok } Available |
| [Pydantic AI](pydantic-ai.md) | `Tool.from_schema`, `requires_approval` | `resolve_deferred` (edits via `override_args`) | :material-check-circle:{ .status-ok } Available |
| [LlamaIndex](llamaindex.md) | `FunctionTool` with `fn_schema` (async + `run_sync`) | — | :material-check-circle:{ .status-ok } Available |
| [CrewAI](crewai.md) | `BaseTool`s (`_run` via `run_sync`, `_arun`) | own environment | :material-check-circle:{ .status-ok } Available |
| [CAMEL-AI](camel.md) | `FunctionTool`s via `camel_tools(registry)` | the executor's port decides (no native gate, no call id); own environment | :material-check-circle:{ .status-ok } Available |
| [AG2](ag2.md) | `FunctionTool`s via `ag2_tools(registry, approval=port)` | one-use token gate in tool middleware | :material-check-circle:{ .status-ok } Available |
| [A2A](a2a.md) | `a2a_app`, `a2a_executor`, `agent_card` | `connect_a2a`, `import_a2a` importer; approvals via `input-required` | :material-check-circle:{ .status-ok } Available |
| [AG-UI](ag-ui.md) | `AGUIBridge` event stream (SSE), `agui_endpoint` | approvals as interrupts; resumes ask your port | :material-check-circle:{ .status-ok } Available |

Where your app runs (FastAPI, Flask, Celery, scripts) is covered on
[Sync code and backends](../how-to/sync-backends.md) and the backend guides:
[FastAPI](../backends/fastapi.md), [Flask](../backends/flask.md),
[Django](../backends/django.md), [Celery](../backends/celery.md),
[Scripts](../backends/scripts.md), [Serverless](../backends/serverless.md),
[Queues](../backends/queues.md). Every backend, its attach call and its tests:
[Backends and runtimes](../support-backends.md).

## Install several

`pip install "agent-adapter[all]"` installs every extra that co-installs from wheels without
downgrading another one. `pip check` and every contract test run on that set in the
`contract-all-compatible` nox session. Pinned ranges and test sessions:
[Support matrix](../support-matrix.md).

| In `all` | Own environment, and why |
|---|---|
| `openai`, `gemini`, `anthropic`, `redis`, `otel`, `langchain`, `langgraph`, `mcp`, `openai-agents`, `pydantic-ai`, `llamaindex`, `openapi`, `agent-framework`, `agno`, `haystack`, `smolagents`, `claude-agent`, `autogen`, `server`, `server-wsgi`, `ag2`, `a2a`, `ag-ui`, `sqlalchemy`, `dynamodb` | `crewai`: needs `mcp<2` (the `mcp` extra needs `>=2.3`) |
| | `strands`: needs `mcp<2.2` |
| | `semantic-kernel`: pins `pydantic<2.14` |
| | `adk`: pins `opentelemetry<=1.44`, older than the `otel` extra resolves to |
| | `camel`: needs `mcp<2` (the `mcp` extra needs `>=2.3`) and pins `pydantic<=2.12.0` (google-genai in `all` needs `>=2.12.5`) |
| | `dspy`: pulls in LiteLLM (about 300 MB), which caps `openai<3`, below the `openai` extra; the adapter itself never uses it |

Install an own-environment extra in a separate venv, for example
`pip install "agent-adapter[crewai]"`; each has its own `contract-<target>` nox session.

## Third-party integrations

The `Integration` protocol (`export_tool`, `bind_context`, `bind_approval`, optional
`compile_flow`) plus the `agent_adapter.integrations` entry-point group: a new framework is
supported without changing the adapter. `get_integration(name)` finds built-in and
third-party integrations the same way. See
[Writing an integration](../contributing/writing-an-integration.md).

## Schema per family

Integrations reuse the three schema transforms:

| Transform | Used by |
|---|---|
| OpenAI strict | OpenAI, OpenAI Agents SDK |
| Gemini | Gemini, ADK |
| Anthropic (plain) | Anthropic, MCP |

Details on [Schemas](../concepts/schemas.md).
