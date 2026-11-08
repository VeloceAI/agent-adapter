# Features

What Agent Adapter does, grouped by area, with a link to each guide. Everything is available
since **1.0**; rows marked **1.1** are new in 1.1. Runnable code for each area is in the
[Examples gallery](examples.md). Not sure where to start? See [Which feature do I need?](choose.md).
Planned work is on the [Roadmap](roadmap.md). No agent framework is needed: a FastAPI service
calling an LLM gets tools, approvals, budgets, cost, cache and compression directly.

## Tools, registry and schemas

| Feature | Guide |
|---|---|
| `@registry.tool` decorator; function stays callable | [Tools](concepts/tools.md) |
| `registry.wrap()` for code you cannot edit; bound methods, callable objects, `functools.partial` | [Wrap existing code](get-started/wrap-existing-code.md#methods-classes-and-blocking-code) |
| Default registry: `agent_adapter.tool`, `agent_adapter.wrap` | [Registry](concepts/registry.md#the-default-registry) |
| One function as several tools (different `fixed=`); sync tools run in a worker thread (contextvars copied) | [Wrap existing code](get-started/wrap-existing-code.md#2-wrap-it), [Wrap a legacy class](how-to/wrap-legacy-code.md#3-blocking-sync-code) |
| Shaping: `fixed`, `hide`, `rename`, `result`, `name`, `description` | [Tools](concepts/tools.md#shaping-options) |
| Descriptions from Google docstrings, `Annotated`/`Field()`; `ToolSpec.returns`, `result_schema()` | [Tools](concepts/tools.md#where-the-schema-comes-from), [Schemas](concepts/schemas.md#return-schemas) |
| Policy options stored and type-checked, incl. `cache_scope`, `targets`, `effects`, `idempotent` | [Tools](concepts/tools.md#policy-options) |
| `wrap(..., args_model=...)` escape hatch; `spec_from_schema`, `input_schema` / `output_schema` (tools from a raw JSON schema) | [Tools](concepts/tools.md#bring-your-own-argument-model), [Schemas](concepts/schemas.md#tools-from-a-raw-json-schema) |
| `ToolContext`: `run_id`, `call_id`, `thread_id`, `state`, `extras`, ports; `child()`; `ToolContext.ledger`, `.cache`, `.emit` filled by the executor | [Context](concepts/context.md), [Executor](concepts/executor-policies.md#cache-ledger-and-events) |
| Typed errors under `AdapterError`, each with a stable `kind` | [Errors](concepts/errors.md) |
| `ToolTimeout`, `ToolFailed`, `ApprovalDenied`, `ApprovalPending` raised by the executor; `BudgetExceeded`, `FlowError` | [Executor](concepts/executor.md#one-call), [Hooks](concepts/hooks.md#budgethook), [Flows](concepts/flows.md#inside-a-node) |
| Importers: `import_langchain`, `import_openapi`, `import_mcp` | [LangChain](integrations/langchain.md#import), [OpenAPI](integrations/openapi.md), [MCP](integrations/mcp.md#import) |
| Registry: `add`, `get`, `remove`, `names`, `in`, `len`, iteration; `add_many` (all or nothing), `for_target` | [Registry](concepts/registry.md#methods), [Targets](concepts/registry.md#targets) |
| `subset(names=, tags=, kinds=)` (filters combine with AND); `catalogue()`: Markdown tool list with `prompt_hint` | [Registry](concepts/registry.md#views-per-agent), [Registry](concepts/registry.md#prompt-catalogue) |
| Portable-subset check at registration; `portable=False` opt-out; Stable rendered schemas per provider (pinned by snapshot tests) | [Schemas](concepts/schemas.md#the-portable-subset), [Schemas](concepts/schemas.md#what-each-provider-gets) |
| OpenAI strict, Gemini and Anthropic transforms; defaults written into descriptions (OpenAI strict, Gemini) | [Schema pipeline](architecture/schema-pipeline.md), [Where defaults go](architecture/schema-pipeline.md#where-defaults-go) |
| Strict-mode `null` means "use the default" | [Strict mode](concepts/strict-mode.md) |

## Execution and flows

| Feature | Guide |
|---|---|
| `run`, `run_many(concurrency=)`, `parallel(*coros)`, `as_completed` | [Batches](concepts/executor.md#batches), [Parallel retrieval](how-to/parallel-retrieval.md) |
| `run_plan(steps)`: `depends_on` waves, cycle check | [Plans](concepts/executor.md#plans) |
| `on_error="collect"` / `"fail_fast"`, `ToolResult` | [Batches](concepts/executor.md#batches) |
| Retry with backoff and jitter; timeout; never retry control flow | [Retry](concepts/executor-policies.md#retry-timeout-and-control-flow) |
| Run guards: `once_per_run`, max calls per run, concurrency per `kind` | [Runs and guards](concepts/executor-policies.md#runs-and-guards) |
| `Hook` base class and dispatch (`before_*` / `after_*` / `on_*_error`, short-circuit), run by the executor | [Runtime](concepts/runtime.md#hooks), [One call](concepts/executor.md#one-call) |
| `GuardrailHook`, `RateLimitHook`, `BudgetHook(max_usd)`; **1.1** team budgets per user, team or all, per day or month, shared across processes (SQLite, Redis, SQL) | [Hooks](concepts/hooks.md#guardrailhook), [Budgets](how-to/budgets.md), [Team budgets](how-to/team-budgets.md) |
| **1.1** Guardrails: PII and secret checks with no model (`PIIGuard`), any guard service (`CallableGuard`), Presidio, NeMo Guardrails; block, redact or approve | [Guardrails](concepts/guardrails.md) |
| **1.1** Policy file: access, approvals, guards, budgets and limits per tool, team and environment, for every surface; `agent-adapter policy check` | [Policy file](concepts/policy.md) |
| **1.1** Multi-agent: agents as tools (`agent_tool`, `LoopAgent`), handoffs (`handoff`, `HandoffModel`), flow routing, depth guard | [Multi-agent](how-to/multi-agent.md) |
| `Flow` builder, `Map` / `Join` / `Route` / `max_loops`, `pause`, keyed merges, local runner; **1.1** sub-flows (`SubFlow`) and a per-node cache (`NodeCache`); **1.1** crash-safe runs (a checkpoint per superstep, `recover`, no repeated side effects) and Temporal or DBOS as the engine | [Flows](concepts/flows.md), [Pause and resume](concepts/flow-resume.md), [Sub-flows and the node cache](concepts/flow-subflows.md), [Durable runs](concepts/durable-runs.md), [Temporal or DBOS](how-to/durable-engines.md) |
| `Executor()` uses the default registry; `run` takes the tool function; `run_sync` / `resume_sync` twins; `configure`, `call`, `acall` | [Executor](concepts/executor.md), [Quickstart](get-started/quickstart.md) |
| `LoopRunner`: one warm event loop for sync code in long-lived processes (`agent_adapter.runtime`) | [Runtime](concepts/runtime.md#looprunner-one-warm-loop-for-a-long-lived-process) |
| Compile flows to LangGraph and to ADK `Workflow` (parity with the local runner) | [LangGraph](integrations/langgraph.md#compiled-flows), [ADK](integrations/adk.md#flows) |

## HITL (approval)

| Feature | Guide |
|---|---|
| `ApprovalPort` protocol, `ApprovalRequest`, `ApprovalDecision` types | [Context](concepts/context.md#ports) |
| Approval enforced by the executor (may edit args); one-use decisions bound to run, call, tool and argument digest | [Approval](concepts/approval.md), [Human in the loop](how-to/human-in-the-loop.md#the-one-use-rule) |
| `AutoApprove`, `AutoDeny`, `CallbackApproval(fn)`, `DeferredApproval`, `ApprovalStore` | [Approval](concepts/approval.md#ports) |
| `approval.batch.ask_each`, `PreparedApproval` (two-phase batch approval) | [Approval batches](concepts/approval-batches.md) |
| Durable approval stores: `SQLiteApprovalStore` (one host, many processes), `RedisApprovalStore` (many hosts, TTL), `SQLAlchemyApprovalStore` (Postgres), `DynamoDBApprovalStore` (AWS serverless, TTL); corrupt rows fail closed (`ApprovalStoreCorrupt`) | [Approval stores](concepts/approval-stores.md), [Database stores](concepts/approval-stores-db.md) |
| `executor.resume(approval_id, decision)` in one call; `DeferredApproval.sqlite(path)` | [Resume a parked call](concepts/approval-resume.md) |
| **1.1** Approval memory: similar past decisions on approval cards and `GET /approvals/{id}` (`approval_context`), proposed confirm rules a person accepts (`propose_rules`, `accept`); never approves | [Approval memory](concepts/approval-memory.md) |
| **1.1** Audit log: a signed, tamper-evident record of approvals, budget stops, guard blocks and memory erases; verify, and export to JSONL or CSV | [Audit log](concepts/audit.md) |
| Bridges: LangGraph, ADK, Agents SDK, Pydantic AI, Agent Framework, Agno, Strands, Haystack, Claude Agent SDK, Semantic Kernel | [LangGraph](integrations/langgraph.md#approvals-and-resume-values), [ADK](integrations/adk.md#tools-and-approval), [Agents SDK](integrations/openai-agents.md#approval), [Pydantic AI](integrations/pydantic-ai.md#approval), [Agent Framework](integrations/agent-framework.md#approval), [Agno](integrations/agno.md#approval), [Strands](integrations/strands.md#approval), [Haystack](integrations/haystack.md#approval), [Claude Agent SDK](integrations/claude-agent.md#approval), [Semantic Kernel](integrations/semantic-kernel.md#approval) |

## Cost and cache

| Feature | Guide |
|---|---|
| `LedgerEntry` per LLM and paid-API call | [Instrumented models](concepts/tool-loop.md#one-instrumented-call) |
| Usage normalisers: OpenAI, Gemini, Anthropic | [Model ports](concepts/models.md#usage) |
| Versioned price table `Prices.load(dict|path)`; **1.1** context windows and reply ceilings for known models (`ModelCaps.prompt_window()`) | [Cost](concepts/cost.md#prices), [Context compression](concepts/context-compression.md) |
| Prices for your own models: `*` patterns, `free`, `per_call` and `per_second`, `Prices.empty()` | [Cost](concepts/cost.md#your-own-models) |
| Sinks: memory, callback, JSON lines; `ledger.summary(run_id)` | [Cost](concepts/cost.md#ledger) |
| `CachePort`, `MemoryCache`, `RedisCache` (extra) | [Cache](concepts/cache.md#backends) |
| Cache keys, versioned JSON envelopes, `ToolResultCache` | [Runtime](concepts/runtime.md#cache-keys) |
| Tool-result cache used by the executor (`cache_scope`: run / thread / global) | [Executor](concepts/executor-policies.md#cache-ledger-and-events), [Caching](how-to/caching.md) |
| LLM response cache | [Cache](concepts/cache.md#llm-response-cache) |
| `PromptLayout(static=, dynamic=)` | [Cache](concepts/cache.md#prompt-layout) |
| Provider prefix-cache plug-ins (OpenAI, Gemini, Anthropic) | [Cache](concepts/cache.md#provider-plug-ins) |

## Models

| Feature | Guide |
|---|---|
| `LLMRequest`, `LLMResponse`, `Message`, `ToolDef` types; `LLMPort` protocol | [Context](concepts/context.md#ports) |
| `LLMPort.generate(request)` and `stream(request)` implementations; **1.1** partial structured output and tool-call deltas while streaming | [Model ports](concepts/models.md#streaming), [Streaming](concepts/streaming.md) |
| `OpenAIModel`, `GeminiModel`, `AnthropicModel`, `CallableModel(fn)`, `FallbackModel` | [Model ports](concepts/models.md) · [Bring your own model](how-to/bring-your-own-model.md) |
| Bring your own model: `CallableModel` on a sync or async function (sync runs in a thread); `OpenAIModel` on Ollama, vLLM, LM Studio or llama.cpp server | [Bring your own model](how-to/bring-your-own-model.md) |
| **1.1** Model recipes: Amazon Bedrock, Azure OpenAI, Mistral, Groq, OpenRouter as `CallableModel`s with usage, prices and typed errors | [Model recipes](how-to/model-recipes.md) |
| Bring your own embedder: `EmbeddingPort`, `CallableEmbedder` (sync or async fn, batching, per-text cache, one ledger entry and trace events per call, OTel spans) | [Embeddings](concepts/embeddings.md) |
| **1.1** Memory: `MemoryPort`, `SQLiteMemory` (keyword and vector search), `CallableMemory`, Mem0, Zep, Graphiti; `memory_tools()` scoped by `ctx.user_id`, never by the model; retention, export and per-user erase | [Memory](concepts/memory.md) |
| **1.1** Retrieval (RAG): `retriever_tool` over any `RetrieverPort` (callable, embedding, LlamaIndex, LangChain); citations; host-set scope filters re-checked on every result | [Retrieval (RAG)](concepts/retrieval.md) |
| **1.1** Sandboxed code: `code_tool` over Docker, a hosted sandbox (`CallableSandbox`: E2B, Modal) or a dev-only local process; approval always required; time, output and network limits | [Sandboxed code](concepts/sandbox.md) |
| `InstrumentedModel`, `executor.model()` | [Instrumented models](concepts/tool-loop.md) |
| `run_tool_loop(model, registry, messages, max_steps)` | [Tool loop](concepts/tool-loop.md#the-tool-loop) |
| Context compression with your own models: `CompressionPolicy` per call (`LLMRequest(compression=)`), `CompressingModel`, `run_tool_loop(compression=)`; trim, cut, shrink tool results, de-duplicate, rank, summarise; `ModelContextOverflow` retried once | [Context compression](concepts/context-compression.md), [Compress a long context](how-to/compress-long-context.md) |
| `compress_for(messages, policy)`: every framework's own message list back in its own type (LangChain, OpenAI, Anthropic, Gemini and every agent framework; the Claude Agent SDK compacts on its own); `langgraph_compressor` and per-framework hooks | [Compression by framework](integrations/compression-by-framework.md) |

## Integrations

| Feature | Guide |
|---|---|
| OpenAI, Gemini, Anthropic renderers (honour `targets`) | [OpenAI](integrations/openai.md), [Gemini](integrations/gemini.md), [Anthropic](integrations/anthropic.md) |
| genai automatic function calling callables, `run_function_calls` | [Gemini](integrations/gemini.md#automatic-function-calling-afc) |
| LangGraph, Google ADK 2.x | [LangGraph](integrations/langgraph.md), [ADK](integrations/adk.md) |
| LangChain export and import, OpenAPI importer | [LangChain](integrations/langchain.md), [OpenAPI](integrations/openapi.md) |
| MCP server and importer (annotations, pending approval, client approval round trip); tasks extension, URL approval links, client OAuth (CIMD) | [MCP](integrations/mcp.md), [MCP tasks and OAuth](integrations/mcp-tasks.md) |
| OpenAI Agents SDK, Pydantic AI, LlamaIndex, CrewAI | [Agents SDK](integrations/openai-agents.md), [Pydantic AI](integrations/pydantic-ai.md), [LlamaIndex](integrations/llamaindex.md), [CrewAI](integrations/crewai.md) |
| Microsoft Agent Framework, Agno, Strands Agents, Haystack | [Agent Framework](integrations/agent-framework.md), [Agno](integrations/agno.md), [Strands](integrations/strands.md), [Haystack](integrations/haystack.md) |
| Claude Agent SDK, Semantic Kernel, DSPy, smolagents, AutoGen | [Claude Agent SDK](integrations/claude-agent.md), [Semantic Kernel](integrations/semantic-kernel.md), [DSPy](integrations/dspy.md), [smolagents](integrations/smolagents.md), [AutoGen](integrations/autogen.md) |
| CAMEL-AI (the executor's port decides `confirm` tools), AG2 1.x (optional token gate in tool middleware) | [CAMEL-AI](integrations/camel.md), [AG2](integrations/ag2.md) |
| A2A: `a2a_app`, `a2a_executor`, `agent_card`, `connect_a2a`, `import_a2a`; approvals via `input-required`, resumed only through the port or a one-use token | [A2A](integrations/a2a.md) |
| AG-UI bridge: stream runs to AG-UI UIs (CopilotKit); interrupt/resume HITL through the `ApprovalPort` | [AG-UI](integrations/ag-ui.md) |
| `attach(target_or_name, registry)`: one call for any installed integration; never takes an approval; attach to plain code, web apps, workers, frameworks, other languages | [Attach anywhere](how-to/attach-anywhere.md) |
| `agent-adapter` CLI: `init` project templates, `doctor` environment check (**1.1**: optional backends too); **1.1** framework templates (pydantic-ai, adk, agno, strands, crewai) and `migrate` | [CLI](reference/cli.md), [The migrate command](migration/migrate-command.md) |
| `Integration` protocol and entry points (third-party) | [Writing an integration](contributing/writing-an-integration.md) |
| `all` extra: every integration that co-installs cleanly | [Integrations](integrations/index.md#install-several), [Support matrix](support-matrix.md) |

## Cross-language and backends

| Feature | Guide |
|---|---|
| MCP stateless Streamable HTTP app | [Cross-language: MCP](cross-language/mcp.md) |
| HTTP tool server: routes, RFC 9457 problems, auth hooks, idempotency, SSE events, cost | [HTTP server](cross-language/http-server.md), [Security model](cross-language/http-server.md#security-model) |
| WSGI app, `/openapi.json`, server CLI | [HTTP server](cross-language/http-server.md#production-settings), [OpenAPI](cross-language/openapi.md) |
| TypeScript client: every server route, typed RFC 9457 errors, safe retries with `Idempotency-Key`, SSE events with reconnect, Vercel AI SDK adapter | [TypeScript client](cross-language/typescript-client.md) |
| Async apps (FastAPI, Django async views): `await executor.run(...)`; `run_sync()` for Flask, Django under WSGI, Celery, RQ, notebooks; `LoopLocal` clients | [Sync code and backends](how-to/sync-backends.md), [Runtime](concepts/runtime.md) |
| Queue and serverless handlers (`agent_adapter.handlers`): `tool_handler` (Lambda: SQS partial-batch failure, FIFO order, EventBridge, direct), `handle_envelope` (any broker); idempotent ids from delivery ids; messages never approve | [Serverless](backends/serverless.md), [Queues](backends/queues.md) |
| Backend guides: FastAPI, Flask, Django, Celery, scripts, data pipelines, other web frameworks, notebooks and apps | [FastAPI](backends/fastapi.md), [Flask](backends/flask.md), [Django](backends/django.md), [Celery](backends/celery.md), [Scripts](backends/scripts.md), [Pipelines](backends/pipelines.md), [Other web frameworks](backends/other-asgi.md), [Notebooks](backends/notebooks.md) |
| `/readyz` readiness probe; Docker and Kubernetes recipe | [Deploy](backends/deploy.md) |
| Connect anywhere recipes (12 stacks) | [Connect anywhere](connect/index.md) |

## Observability

| Feature | Guide |
|---|---|
| Standard `logging` (`agent_adapter.*` loggers); `LoggingHook` (structured, redacts secrets) | [Logging](concepts/errors.md#logging), [Hooks](concepts/hooks.md#logginghook) |
| Event catalogue (`EventType`, `make_event`), `ToolContext.emit`; events emitted by the executor | [Context](concepts/context.md#events), [Executor](concepts/executor-policies.md#cache-ledger-and-events) |
| Event sinks and OpenTelemetry; run viewer: `record()` then `agent-adapter ui` (navigable flow graph with replay, parallel-call timeline, events, state, compare, approvals, offline export), `Flow.graph()`, `Flow.to_mermaid()`; chat with any agent or the built-in tool loop (`agent-adapter ui --agent`, `serve(...)`): streaming, tool and approval cards, state, files, replay, eval cases; **1.1** search, tags and OTLP import in the run viewer | [Tracing](concepts/tracing.md), [See your runs](how-to/see-your-runs.md), [Navigate the graph](how-to/navigate-the-graph.md), [Chat with your agent](how-to/chat-with-your-agent.md), [Search and tag runs](how-to/search-your-runs.md) |
| **1.1** Chat memory across chats: `serve(memory=, user_id=, memory_save=)` / `ui --memory sqlite:PATH --user NAME`; recall each turn (`session.memories`), opt-in saving, a Memory tab to see and delete records | [Remember users across chats](how-to/chat-memory.md) |
| **1.1** Traces to Langfuse, Arize Phoenix, OpenLLMetry or any OTLP backend over `OTelSink` | [Send traces](how-to/send-traces.md) |

## Evals, optimiser and documentation

| Feature | Guide |
|---|---|
| **1.1** Evals: run cases on any agent, graders (exact, regex, JSON schema, tools, cost, latency, LLM judge), JUnit/Markdown/JSON reports, regressions, `agent-adapter eval`, a pytest plugin, an Evals tab in the viewer; a production checklist; recorded runs become eval cases (viewer button, CLI, Python); a cost advisor picks a cheaper model per flow node or stage that keeps the score | [Evaluate your agent](how-to/evaluate-your-agent.md), [See your evals](how-to/see-your-evals.md), [Production checklist](guides/production-checklist.md), [Evals from production](how-to/evals-from-production.md), [Cut model costs per step](how-to/cut-model-costs.md) |
| Offline Propose → Probe → Compose loop; outputs a reviewable diff; `CallableHost` makes any agent optimisable | [Optimiser](optimiser.md), [Optimise your agent](how-to/optimise-your-agent.md) |
| Documentation with an API reference; every code snippet is a runnable, tested example; one guide for AI coding agents generated from the code (`AGENTS.md`, `llms-full.txt`, `agent-adapter guide`) | [Home](index.md), [API reference](reference.md), [Quickstart](get-started/quickstart.md), [For AI coding agents](get-started/for-ai-agents.md) |
| How-to guides (legacy classes, parallel retrieval, HITL, budgets, caching, sync, testing) and tutorials (a research agent, migrating from LangGraph); the golden path: a FastAPI desk with approvals, budgets, guards, memory, RAG, a durable flow and evals in CI | [Wrap a legacy class](how-to/wrap-legacy-code.md), [Testing tools](how-to/testing-tools.md), [Research agent](tutorials/research-agent.md), [Migrate from LangGraph](tutorials/migrate-langgraph.md), [Tutorial: the golden path](tutorials/golden-path.md) |
| Cookbook, guides (security, performance, observability, versioning), migration, troubleshooting, glossary; contributing (integrations, performance tests, releases); a threat model and the API stability policy | [Cookbook](cookbook/index.md), [Security](guides/security.md), [Migration](migration/index.md), [Troubleshooting](troubleshooting.md), [Glossary](concepts/glossary.md), [Writing an integration](contributing/writing-an-integration.md), [Performance tests](contributing/performance-tests.md), [Cutting a release](contributing/release.md), [Threat model](guides/threat-model.md), [API stability policy](contributing/stability.md) |
