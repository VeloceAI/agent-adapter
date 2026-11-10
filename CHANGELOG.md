# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `dspy_compress_prompt(prompt, policy)` in `agent_adapter.integrations.dspy`, and the `"dspy"`
  compression format for the lm15 messages in a DSPy 3.5 `Prompt`. It takes what
  `Adapter.format` returns on DSPy 3.5 or 3.4 and gives back the same kind.

### Fixed

- DSPy 3.5 support. Tools already ran unchanged; the compression adapter example failed
  because DSPy 3.5 adapters return a `Prompt`, not chat dicts. The examples and contract
  tests now cover DSPy 3.5 (XMLAdapter default, the `tool_calls` and `submit` ReAct) and 3.4.

## [1.1.0] - 2026-11-05

The first public release: open source under Apache-2.0 and published on PyPI.

### Added

- **Guardrails.** `PIIGuard`: a dependency-free guard hook for emails, phone numbers, cards
  (Luhn), IBANs (mod-97) and API keys or secrets. It blocks, redacts tool results and model
  replies, or sends flagged tool calls to one-use approval (`confirm=guard.confirm`). It fails
  closed and never logs content.
- `CallableGuard(fn)` for any guard service; `PresidioGuard` (extra `presidio`) and
  `NemoGuard` (extra `nemo`); `ContentGuard`, `TextGuard` and `GuardVerdict` to write your own.
- **Multi-agent.** `agent_tool(agent, ...)`: any agent (a `LoopAgent`, a `fn(task, ctx)`, or a
  chat agent via `chat_agent_task`) becomes a registry tool that runs in the calling run. Cost,
  budgets, limits, approvals and events roll up, with `parent_id` links. `AgentDepthExceeded`
  caps nesting (`max_depth`, default 3).
- `handoff(...)` and `HandoffModel`: hand the conversation to another agent; its answer ends
  the tool loop.
- **Sub-flows.** `SubFlow(flow, input=, into=)` runs another flow as one node. Fields map in
  and out under the parent's merge rules. A pause or approval inside the child pauses the
  whole run, and resume continues from where it stopped. Sub-flow cycles and field types are
  checked at `validate()`. Mermaid draws a subgraph, and the run viewer opens a sub-flow node
  (`⊞ open`, keys `o` / `u`).
- **Node cache.** `Flow.node(..., cache=NodeCache(ttl=, key=, reads=, scope=))` reuses a
  node's updates from the executor's cache port. Entries are signed with the runner's
  checkpoint key. A node with a pause gate or a tool that needs approval is never cached. A
  hit emits the new `node.cached` event.
- **Context windows.** Price table `2026-11-05`: `context_window` and `max_output_tokens` caps
  (with the vendor page in a new `source` field) for every cited OpenAI, Gemini and Claude
  row, so percent compression policies need no setting for them.
- Price rows for gpt-5.6-sol/-terra/-luna, gpt-5.5, gpt-5.4-mini, gpt-6-astra, gpt-6.1-sol,
  gpt-6.1-terra, gpt-6-luna, gemini-3.6-flash, gemini-3.5-flash-lite, gemini-3.1-flash-lite,
  and Claude Fable 5.1, Opus 5.5, Sonnet 5.5 and Haiku 5.5 (with 5m and 1h cache-write
  prices).
- `ModelCaps.max_output_tokens` and `ModelCaps.prompt_window()`; percent compression applies
  to the window less the reply ceiling.
- **Model recipes.** Amazon Bedrock (Converse), Azure OpenAI, Mistral, Groq and OpenRouter as
  `CallableModel`s, with usage, prices and typed errors (docs: Model recipes).
- **Send traces.** Langfuse, Arize Phoenix and OpenLLMetry over the OTel sink (docs: Send
  traces).
- `agent-adapter doctor` checks optional backends from a pluggable list: Presidio and NeMo
  guards, the OpenTelemetry SDK and OTLP exporters.
- A spec-drift test keeps the docs, the LLM guide and the public API in agreement.
- **Evals.** `agent_adapter.evals`: `run_eval(target, cases, graders)` runs each case as its own
  run on your executor (approvals, budgets, ledger, events, recorder) and returns an
  `EvalReport` with per-case, per-tag and per-grader totals, cost and timing, `compare()` for
  regressions, and `to_junit()`, `to_markdown()`, `to_json()`. Targets: any `fn(input)`, a chat
  agent `fn(message, session)`, `FlowTarget`, `ToolLoopTarget`. Graders: `exact`, `contains`,
  `regex`, `json_schema`, `tools_called`, `max_cost`, `max_latency`, `code_check`, `all_of`, and
  `LLMJudge(model, rubric)` on your own model. Cases saved by the chat page load unchanged.
- `agent-adapter eval TARGET CASES`: `--grader`, `--min-score`, `--junit`, `--markdown`, `--json`,
  `--baseline`; exits 1 below the score, on any case error, or on a regression.
- Pytest plugin: `@pytest.mark.eval_cases("cases.jsonl")` runs a test once per case
  (`eval_case` fixture); `assert_eval` fails with each grader's reason and the run id.
- **Evals in the viewer.** `agent-adapter ui` has an Evals tab: every saved report with its
  score, pass/fail, cost and time, per-tag and per-grader totals; each case with its input,
  expected value, output, every grader's score and reason, and a link to its run; two
  reports side by side with regressions marked (`EvalReport.compare`). Save reports with
  `run_eval(..., save_to=)`, `save_report(report, folder)` or `agent-adapter eval --save`;
  `agent-adapter ui --evals FOLDER` picks the folder (default `.agent_adapter/evals`).
- Eval case results keep the case's `input` and `expected` text.
- Grader specs on the command line: `regex=`, `json-schema=`, `tools=`, `max-cost=`,
  `max-latency=` (`module:attr` is always your own grader). `--min-pass-rate`, `--no-content`
  (`run_eval(capture_content=False)`) and `LLMJudge(trust_passed=)`.
- **Memory.** `agent_adapter.memory`: `MemoryPort`, `SQLiteMemory` (one file, FTS5 keyword
  search with a LIKE fallback, optional vector search through any `EmbeddingPort`),
  `CallableMemory`, and lazy `Mem0Memory`, `ZepMemory` and `GraphitiMemory` (extras `mem0`,
  `zep`, `graphiti`). `memory_tools()` adds `remember`, `recall` and `forget` tools whose scope
  comes from the context, never from the model. Searches never cross users.
- Memory retention and privacy: `retention_days` and `expires_at` are enforced on read and
  purged on write; `export(scope)` serves data requests; `delete(scope=MemoryScope(user_id=...))`
  erases a user.
- `ToolContext.user_id`: the authenticated end user, set by the host (`Executor.context(user_id=)`).
- Events `memory.added`, `memory.searched`, `memory.deleted` and `memory.exported` (counts only).
- **Retrieval (RAG).** `agent_adapter.retrieval`: `RetrieverPort`, `Document` (id, text, score,
  source), `CallableRetriever`, an in-memory `EmbeddingRetriever` (`scope_key` isolation),
  `LlamaIndexRetriever` and `LangChainRetriever`, and `retriever_tool`: a registry tool whose
  results carry sources for citations, with host-set `k` and filters that every result is
  re-checked against. Events `retrieval.searched` and `retrieval.failed` (counts and ids; texts
  only with `capture_content`).
- `agent-adapter doctor` also reports the memory extras and the LlamaIndex and LangChain
  retrievers.
- **Chat memory across chats.** `serve(..., memory=, user_id=, memory_save="none"|"summary"|"facts",
  memory_every=, recall_k=)` and `agent-adapter ui --memory sqlite:PATH --user NAME --memory-save
  ... --memory-every N --recall K`. Each turn recalls the user's records from other chats as
  `session.memories` (the built-in loop: a system note). Saving is opt-in: by the model, or by
  the agent's `save_memories`. `ctx.user_id` is the chat's user, chats are kept per user, and
  the inspector has a Memory tab to see and delete the user's records.
- **Approval memory** in `agent_adapter.approval`: `approval_context` shows similar past
  decisions (counts, deciders, outcomes; never argument values or secret fields) on chat
  approval cards, in the Runs approvals desk, and in `GET /approvals/{id}` for approvers (the
  TypeScript `Approval.context`). `propose_rules` suggests `field == value` / `field <= limit`
  confirm rules; `accept(rule, accepted_by=...)` gives the code and config to commit, and the
  `confirm=` callable. Nothing is ever approved from memory.
- Viewer: `memory.*` events show as `memory` spans in the timeline; run summaries carry a
  `memory` count, and a top-level `retrieval.failed` counts as a failure.
- `agent_adapter.chat`: the chat agent contract (`ChatMessage`, `ChatSession`, `Attachment`,
  `ChatAgent`, `ChatReply`) in one place, used by the chat page and evals. `agent_adapter.viewer`
  and `agent_adapter.viewer.chat` re-export the same classes.
- Memory scopes: `thread_scope` is opt-in; `scope_from_context` no longer gives anonymous
  callers memory; SQLite erasure leaves no text in the file.
- **Crash-safe runs.** `FlowRunner(checkpoints=store, checkpoint_key=KEY)` saves a sealed
  checkpoint after every superstep and logs every tool call. `runner.recover(flow, run_id)` and
  `runner.recover_all(flow)` continue a crashed run on any worker, and calls that finished replay
  instead of running again. A call left in doubt runs again only for an `idempotent=True` tool or
  with `rerun_in_doubt=True`; otherwise `RecoveryNeedsDecision` is raised. Two workers never both
  continue a run (`CheckpointConflict`), and a changed graph refuses to recover.
  `idempotency_key()` gives each call a key that is the same on every worker and recovery.
- `run_tool_loop(..., run_id=, checkpoints=, checkpoint_key=)` and `recover_tool_loop(...)` do
  the same for the tool loop.
- Checkpoint stores in `agent_adapter.exec.checkpoints`: `MemoryCheckpointStore`,
  `SQLiteCheckpointStore`, `RedisCheckpointStore` (extra `redis`), `SQLAlchemyCheckpointStore`
  (extra `sqlalchemy`), and the `CheckpointStore` protocol. New errors `CheckpointNotFound`,
  `CheckpointConflict`, `CheckpointStoreCorrupt`, `RecoveryNeedsDecision`; new events
  `checkpoint.saved` and `run.recovered`; the viewer's run summary carries `recovered: true`.
- **Durable engines.** Run a `Flow` on Temporal (extra `temporal`) or DBOS (extra `dbos`). Each
  superstep is one activity or step through your executor, so approvals, budgets, cache, ledger
  and events apply; a retried step replays finished tool calls from the checkpoint store, and
  runs carry a `RunIdentity`. Approvals stay in your approval store: a Temporal update or
  signal, or a DBOS message (`notify`), only wakes the run to re-check the store, and the stored
  decision is used once; a timeout records a denial. `agent_adapter.integrations.temporal`
  (`flow_worker`, `start_flow`, `notify`, `pending`, `result`) and
  `agent_adapter.integrations.dbos` (`DbosFlows`), over the shared `DurableFlows`.
  `FlowRunner.inspect()` and `FlowRunner.on_superstep` support them.
- **Sandboxed code.** `agent_adapter.sandbox`: `SandboxPort` and `SandboxResult`;
  `DockerSandbox` (the docker CLI, no SDK: `--network none`, read-only root, `--cap-drop ALL`,
  no-new-privileges, a non-root user, pids/memory/cpu limits, a noexec tmpfs, removed on
  timeout); `CallableSandbox` for E2B, Modal, Daytona or any service; `LocalSubprocessSandbox`
  for development only (refuses without `allow_unsafe_local=True`, warns on every run); and
  `code_tool`, a registry tool that always needs approval (no opt-out), with host-set time,
  output, network and file limits. Produced file names are checked on the host. New events
  `sandbox.started`, `sandbox.finished` and `sandbox.failed` (sizes and exit codes; code and
  output only with `capture_content`). `agent-adapter doctor` reports whether Docker can run.
- **Team budgets.** `BudgetHook(scope="user"|"principal"|"team", per="day"|"month",
  scope_key=, store=)` caps spend per user, team or everything, per UTC day or month, shared
  across processes: it reserves an estimate atomically before each model call, settles the real
  cost after and releases on failure. Stores: `MemoryBudgetStore`, `SQLiteBudgetStore`,
  `RedisBudgetStore`, `SQLAlchemyBudgetStore` (`agent_adapter.hooks.builtin.budgets`).
  `GET /budget` and `/budget/{scope}` on the HTTP server (own scope; approvers any key),
  `budget.reserved` / `budget.exceeded` events, the viewer marks runs a budget refused, the
  TypeScript client's `budget()`, and doctor rows.
- **Streaming.** `PartialOutput` events carry the structured output parsed so far (OpenAI,
  Gemini, Anthropic, `CallableModel(stream_fn=)`); `run_tool_loop(on_event=)` gets tool-call
  deltas, partial output and `StreamDone`; stream event types are importable from
  `agent_adapter.models`. A chat agent or eval target may yield `ToolCallDelta` between text
  chunks.
- Chat page: tool arguments appear in a draft tool card while the model streams them (masked,
  cut to 2 KB).
- Viewer: search across runs (tool, flow, node, model and error names, run ids, `status:`,
  `tag:`, captured content) and run tags stored in a `tags.<run key>.json` sidecar per run, set
  from the run header. `agent-adapter ui --import traces.json` and
  `agent_adapter.viewer.import_otlp`: OTLP JSON GenAI spans become viewer runs (parsed only,
  size-limited).
- **Framework templates.** `agent-adapter init pydantic-ai | adk | agno | strands | crewai`: a
  starter with registry tools, a `confirm=True` tool behind an approval port, a budget and the
  run recorder. It runs offline; set `AGENT_MODEL` to run the agent.
- **`agent-adapter migrate PATH`.** A read-only scan (Python `ast`; it never imports or writes)
  for LangChain, CrewAI, OpenAI Agents, ADK, Pydantic AI, LlamaIndex, smolagents, Agno and
  Strands tools and raw OpenAI or Anthropic schemas; `--diff` prints a suggested patch to
  `@registry.tool` / `registry.wrap` with the matching bridge; `--framework` filters.
- **Policy file.** `agent_adapter.policy.load_policy(path)` reads one YAML, TOML or JSON file (or
  a dict) that sets, per tool, team, user and environment (`AA_ENV`): allowed tools
  (`default: deny`), approvals (`confirm: true` or a safe condition such as `amount > 100`),
  guards, budgets, rate limits, run limits, memory retention and save rules, and sandbox
  requirements. `policy.apply(executor)` builds them from the existing hooks and approval path,
  so the tool loop, every integration, MCP and the HTTP server obey it. A policy only makes
  things stricter: it never removes an approval, unknown keys fail at load with the file, line
  and key, and the two relaxations need `allow_relax=True`. `policy.explain(tool, ctx)`, the
  viewer's Tools tab and `agent-adapter policy check FILE` show the effective rules.
  `Executor(policy=)` is the new seam; it is enforced whatever `replace(...)` changes, and
  `Executor.hooks` lists only the hooks you passed.
- **Audit log.** `agent_adapter.audit`: `audit(executor, folder=, key=)` keeps a signed,
  hash-chained JSONL log of approvals (requested, decided, consumed, denied, expired), guardrail
  blocks and redactions, budget stops, memory adds, deletes and exports, sandbox runs, recovered
  runs and your own `audit.<name>` events (recorded as `app.<name>`), never arguments or
  content; records are written on a background thread. `verify()` detects edits, deletions,
  reordering and truncation, and `verify --anchor` catches rollbacks; `export()` and `agent-adapter audit verify|export` give JSONL or
  CSV; `CallableAuditStore` ships records to a SIEM or bucket. `fail_closed=True` blocks calls
  before their side effect when a record cannot be written; `verify`, `read_entries` and
  `export_to` stream. New events `approval.expired` and `guardrail.redacted`;
  `approval.decided` carries `edited`; memory events carry `scope_kind`. Redis and DynamoDB
  TTL expiries are recorded at resume.
- **Adoption pack.** A golden-path FastAPI support desk (`examples/golden_path/`): confirmed
  refunds with a SQLite approval store, a daily budget per user, PII guards, memory across
  chats, cited help answers, a crash-safe flow, recorded runs, and eval cases run by
  `agent-adapter eval` as the CI gate (docs: Tutorial: the golden path).
- `scripts/bench_overhead.py` measures the executor's per-call overhead over a direct call, per
  feature; measured numbers are in the Performance guide.
- A threat model page (assets, trust boundaries, what the adapter enforces and what it does
  not) and an API stability policy for 1.x (public surface, semver rules, deprecation,
  experimental markers).
- **Production → evals loop.** `agent_adapter.evals.cases_from_runs` and
  `agent-adapter eval cases-from-runs RUNS_DIR --out cases.jsonl` turn recorded runs (captured
  content only; others are skipped with the reason) into eval cases with `source_run`; the Runs
  view gets "Save as eval case" and, on failed runs, "Turn into regression case", appending to
  the chat's `eval_cases.jsonl`.
- Opt-in content on native events: a sink with `wants_content = True` (the run recorder under
  `capture_content=True`) gets tool arguments and results, model messages, and a flow's input
  and final state; other sinks get none, and content is not built for them. Content is taken
  after guards (arguments on `tool.finished`, messages on `llm.finished`), secret-named keys
  and secrets and PII inside text are masked, `confirm` tools' arguments are never sent, and
  their results need `capture_confirm_content=True`.
- **Cost advisor.** `agent_adapter.evals.advisor.advise(target_factory, cases, graders,
  candidates=, baseline=)` runs your evals on the baseline model, then tries each cheaper
  candidate in one flow node or stage at a time, and reports the cheapest model per step that
  keeps the score (`min_score_ratio`, default 0.98), with cost and latency deltas from the
  ledger. `max_spend=` caps the candidates' spend and stops early. `report.mapping` holds only
  verified advice and is never applied for you: `apply_mapping` builds a `StepModels` port for `FlowRunner(model=...)` or
  `run_tool_loop` (docs: Cut model costs per step). Ledger entries
  of model calls made inside a flow node carry `metadata["node"]`.
- Docs: a production checklist (approvals, budgets, guardrails, tracing, evals in CI,
  caching, logs, viewer and server security).

### Fixed

- The ADK starter template runs offline without `google-genai`; `adk_tools()` imports it only
  when ADK asks for a tool declaration.
- The viewer keeps runs started within one clock tick in order (newest first), and rotation
  deletes the oldest of them.
- `SQLAlchemyBudgetStore.aclose()` disposes of the running loop's engine, like the approval
  and checkpoint stores.

### Changed

- Licensed under Apache-2.0 (`LICENSE`, `NOTICE`; PEP 639 `license = "Apache-2.0"`). The
  package installs from PyPI: the `Private :: Do Not Upload` classifier and the
  internal-index install steps are gone. The TypeScript client is Apache-2.0 too (still
  `private`).
- `agent-adapter doctor`: public PyPI or a mirror as the index is `ok`; an
  `extra-index-url` or `UV_EXTRA_INDEX_URL` is a warning, no longer an error.
- Live-provider checks (the OpenAI usage formula, Anthropic structured output, the
  `google-genai` wire keys) were run by the maintainer before release.

- `Prices.default().version` is now `2026-11-05`; `claude-` stays caps-only for Claude ids
  without a row.
- The LangGraph and ADK flow compilers raise `FlowCompileError` for `SubFlow` nodes.
- The `dev` extra adds `pytest-xdist` and `pytest-testmon`.
- The optimiser's `EvalCase`, `Score`, `load_cases` and `optimiser.graders` now live in
  `agent_adapter.evals`; the optimiser names are re-exports, so existing imports keep working.
- `agent-adapter doctor` help names the optional-backend checks.
- `graders.all_of` fails closed: `passed=False` when a part errors or does not grade.
- `agent-adapter ui --chat-dir` also works without chat (it is where Runs-view eval cases go).
- The eval pytest plugin is opt-in (`-p agent_adapter.evals.pytest_plugin`, or
  `pytest_plugins` in a `conftest.py`), so installing the package changes no pytest run.

## [1.0.0] - 2026-10-31

First stable release. Every feature listed on the Features page is available.
The suite runs offline; the points that still need live API keys are listed under Notes.

### Highlights by area

- **Tools, registry, schemas.** `@registry.tool` / `registry.wrap()` on any callable,
  shaping, tools from raw JSON schemas, a portable-schema check, OpenAI (strict), Gemini and
  Anthropic transforms, typed errors with stable kinds.
- **Execution.** One `Executor` pipeline for every call: run guards, validation, hooks,
  approval, tool-result cache, per-run locks and per-kind caps, timeout and retry, ledger
  and events; `run_many`, `run_plan`, `parallel`, `as_completed`, `run_sync`.
- **Hooks.** `LoggingHook` (redacting), `BudgetHook`, `GuardrailHook`, `RateLimitHook`.
- **Approval.** `confirm` rules, `AutoApprove` / `AutoDeny` / `CallbackApproval` /
  `DeferredApproval`, stores bound to tool and argument digest, two-phase batch approval.
  Decisions are used once; a framework's own "approved" signal never counts alone.
- **Flows.** A portable `Flow` spec (map, join, route, pause, keyed merges) with a local
  runner, sealed checkpoints and resume; compiles to LangGraph and ADK with parity.
- **Cost.** Versioned price table, usage normalisers, a priced `Ledger` with sinks and
  summaries; prices for your own models (patterns, `free`, per call, per second).
- **Cache.** Memory and Redis backends, tool-result and LLM-response caches,
  `PromptLayout`, provider prefix-cache plug-ins.
- **Models.** OpenAI, Gemini, Anthropic, callable and fallback ports with streaming;
  `InstrumentedModel`; `run_tool_loop`. Bring your own model or embedder
  (`CallableModel`, `CallableEmbedder`); the adapter wraps it.
- **Imports.** A lazy 55-name top level, `agent_adapter.errors`, and one public home per
  name.
- **Context compression.** A `CompressionPolicy` shrinks long histories before each model
  call with your own models; free and AI steps; every framework's own message type in and
  out (`compress_for`); one retry on `ModelContextOverflow`.
- **Integrations.** 25 built-in integrations (3 model-API renderers, the OpenAPI importer,
  MCP, A2A, AG-UI and 18 frameworks) plus the `Integration` protocol, entry points and
  `attach()`.
- **Cross-language.** MCP server and importer, HTTP tool server (ASGI, WSGI, CLI,
  `/openapi.json`) with principal-scoped ids and idempotency, TypeScript client.
- **Observability.** Event catalogue, event sinks, OpenTelemetry GenAI spans and metrics.
- **Run viewer and chat.** `record()` writes every run to a local folder; `agent-adapter ui`
  shows a navigable flow graph with replay, a timeline of parallel calls, events, state,
  costs and pending approvals on 127.0.0.1, for any framework or plain code, offline. With
  `--agent` or `--model` (or `serve(...)` from Python) the same page chats with your agent:
  streamed answers, tool and approval cards, and a live inspector per turn.
- **Brand.** The display name is "Agent Adapter" (package `agent_adapter`, command
  `agent-adapter`). The viewer, chat page, exports and docs site show the logo as the
  browser-tab icon; the docs site loads no web fonts and no script from another host.
- **Optimiser.** Offline Propose → Probe → Compose loop that writes a reviewable patch.
- **Docs.** Attach-anywhere guide, how-to guides, backend guides (FastAPI, Flask, Django,
  Celery, scripts), tutorials, a release guide, and an API reference for every public
  module; every snippet is a tested file under `examples/`.

### Added

- `Executor()` uses the default registry; `Executor.run` takes the `@tool` function, a name
  or a spec; sync twins (`SyncTwins`: `run_sync`, `run_call_sync`, `run_many_sync`,
  `resume_sync`); `configure`, `call`, `acall`; `Executor.resume` / `resume_sync`;
  `DeferredApproval.sqlite(path)`; top-level `CallableModel`, `FallbackModel`; fix-it notes
  on common errors and "did you mean" hints for top-level typos.
- `attach(target_or_name, registry)`: one call for any installed integration (entry-point
  discovery; never takes an approval option). The `agent-adapter` command (also
  `python -m agent_adapter.cli`): `init` project templates and `doctor`.
- `LoopRunner` (`agent_adapter.runtime`): one warm loop for sync code in long-lived
  processes; fork-safe `LoopLocal`; `/readyz` readiness probe and `server.Readiness`; a
  clear `ToolUsageError` when pickling `Executor`, `Registry` or `Ledger`.
- `agent_adapter.handlers`: `tool_handler(executor)` for AWS Lambda (SQS batches with
  `batchItemFailures`, FIFO-safe, EventBridge, direct invoke) and
  `handle_envelope(executor, payload)` for any queue. Handlers share one `LoopRunner` by
  default; `runner=RunSync()` gives a fresh loop per invocation.
- `SQLAlchemyApprovalStore` (Postgres/SQLite via SQLAlchemy 2 async; `ON CONFLICT` and
  `UPDATE … RETURNING`) and `DynamoDBApprovalStore` (conditional writes, TTL), extras
  `sqlalchemy` and `dynamodb`, in `agent_adapter.approval`.
- A2A integration (`agent_adapter.integrations.a2a`, extra `a2a`): `a2a_app`,
  `a2a_executor`, `agent_card`, `connect_a2a`, `import_a2a`; approvals via
  `input-required`, resumed only through the executor's port or a one-use token; in the
  `all` extra (co-install checked with `pip install --dry-run '.[all]'`).
- AG-UI 1.0 bridge (`agent_adapter.integrations.ag_ui`, extra `ag-ui`): `AGUIBridge`
  streams runs as AG-UI events over SSE, turns pending approvals into interrupts and sends
  resume entries to the `ApprovalPort` with one-use tokens; `agui_endpoint` for Starlette
  and FastAPI.
- CAMEL-AI integration (`agent_adapter.integrations.camel`, extra `camel`): registry tools
  as `FunctionTool`s; the executor's port decides `confirm` tools; own environment (pins
  `pydantic<=2.12.0`, which google-genai in `all` rules out).
- AG2 1.x integration (`agent_adapter.integrations.ag2`, extra `ag2`): registry tools as
  `FunctionTool`s with an optional token-based approval gate in AG2 tool middleware.
- MCP: the 2026-07-28 tasks extension (`mcp_server(tasks=TaskOptions(...))`): slow calls
  and pending approvals become task handles that a host polls by task id, so `confirm`
  tools complete through tasks-capable hosts that drop `_meta` keys; URL-mode approval
  links (`TaskOptions(approval_url=...)`); `mcp_client(tasks=True)` follows and resumes
  remote tasks; client OAuth with Client ID Metadata Documents (`oauth_client`,
  `mcp_client(auth=...)`).
- Optimiser: `CallableHost` makes any agent optimisable (a plain function, or a LangGraph,
  OpenAI Agents or Pydantic AI agent you wrap); `load_cases` (JSONL/JSON/CSV); graders
  `exact`, `contains`, `regex`, `code_check`, `llm_judge`, `all_of`; `load_harness` and
  `layout_from_tree` to apply a chosen tree; CLI `--model none` (`NullModel`).
- Docs: troubleshooting, "Which feature do I need?", support matrix, cookbook (8 recipes),
  guides (security, performance, observability, versioning), migration (raw tool calling,
  LangChain tools, hand-rolled approval), "Connect anywhere" (Hermes Agent, Letta, Mastra,
  Spring AI, MAF .NET, Genkit, ADK Go, LangGraph.js, Vercel AI SDK, OpenAI hosted MCP,
  Azure AI Foundry, Bedrock AgentCore Gateway), backend guides (serverless, queues,
  pipelines, other web frameworks, notebooks, deploy with Docker and Kubernetes), database
  approval stores, resuming a parked call, optimising your agent, a glossary, an error
  catalogue, `llms.txt`, a reference page per package and the CLI reference; a new logo.
- Docs: how-to guides (attach anywhere, wrap legacy code, parallel retrieval, human in the
  loop, budgets, caching, sync code and backends, testing tools), backend guides (FastAPI,
  Flask, Django, Celery, scripts), tutorials (a research agent, migrating from LangGraph),
  Contributing → Cutting a release; the API reference covers every integration module,
  `runtime.ids`, `models.instrument` / `entries`, `tracing.semconv` and the flow-compiler
  seams (`exec.integration`, `flow.visit`).
- `tests/test_features_doc.py`: every Features row is Available and links an existing page.
- Error catalogue: every error has a stable `kind` and a `retryable` flag and survives
  pickling. New errors: `ToolError`, `ToolTimeout`, `ToolFailed`, `RunLimitExceeded`,
  `GuardrailBlocked`, `ApprovalDenied`, `ApprovalPending`, `ApprovalNotFound`,
  `ApprovalAlreadyDecided`, `BudgetExceeded`, `PlanInvalid`, `FlowError`,
  `FlowDefinitionError`, `FlowCompileError`, `FlowLoopExceeded`, `ModelError` and its
  subclasses (`ModelRateLimited`, `ModelUnavailable`, `ModelTimeout`, `ModelAuthError`,
  `ModelBadRequest`, `ModelContentFiltered`, `ModelOutputInvalid`), `ConfigError`,
  `IntegrationError`, `AuthError`, `Unauthorized`, `Forbidden`, `OptimiserError`.
- Value and message types: `ToolCall`, `ToolResult`, `ErrorInfo`, `Event`, `Usage`,
  `LedgerEntry`, `ApprovalRequest`, `ApprovalDecision` (`Decision`), `ModelCaps`,
  `Message`, `ToolDef`, `OutputSchema`, `LLMRequest`, `LLMResponse` (with `parse()`),
  stream events; ports `CachePort`, `LedgerPort`, `EventSink`, `ApprovalPort`, `LLMPort`,
  `PromptCachePlugin`.
- `ToolContext` fields `thread_id`, `ledger`, `cache`, `sink`, `approval`, plus `emit()`
  and `child()`.
- `ToolSpec` fields `cache_scope`, `targets`, `input_schema`, `output_schema`, `effects`,
  `idempotent` (the first, fourth and last three as policy options) and `tool_def()`.
- `build_spec(..., args_model=...)` (and `wrap`/`@tool`) for a hand-written argument model.
- `spec_from_schema(...)`: tools from a raw JSON schema, validated by a loose model
  (`schema.models.model_from_json_schema`); unset optional arguments are not sent.
- `Registry.add_many` (all or nothing) and `Registry.for_target`; the OpenAI, Gemini and
  Anthropic renderers honour `targets`.
- `run_sync()`, `LoopLocal`, `current_context()` / `use_context()`, `new_id()`.
- `Hook` base class, `ShortCircuit` and the `run_before` / `run_after` / `run_on_error`
  dispatchers.
- Event catalogue (`tracing.EventType`, `REQUIRED_DATA`, `make_event`, `emit`), cache keys
  (`cache.tool_key`, `llm_key`, `gcache_key`), versioned JSON envelopes and
  `ToolResultCache`.
- Extras for every model SDK and framework integration, `all`, nox sessions for each.
- `agent_adapter.exec.Executor`: `run`, `run_spec`, `run_call`, `run_many`, `as_completed`,
  `run_plan` (`Step`, `PlanInvalid`), `parallel`, `context`, `run_scope`, `end_run`,
  `replace`. Every call runs one fixed pipeline: run guards (`once_per_run`,
  `RunLimits.max_calls`, counted per distinct call id), validation, hooks, approval
  (`ctx.approval`, `confirm` rules, edited args re-validated), tool-result cache, a per-run
  readers-writer lock for `parallel_safe=False` tools (re-entrant for nested calls),
  `RunLimits.per_kind` caps, timeout and retry with `Backoff`, ledger entries and trace
  events. Batch methods never raise `ExceptionGroup`.
- `exec.register_control_flow` / `is_control_flow`; `exec.default_executor`,
  `set_default_executor`, `runner_for`.
- Built-in hooks (`agent_adapter.hooks`): `LoggingHook` (one JSON line per start, finish
  and error with run and call ids; redacts secret-looking keys; never logs confirm-tool
  arguments or message content), `BudgetHook(max_usd, on_exceed=, scope="run" | "thread"
  | "global")`, `GuardrailHook(check, on=...)`, `RateLimitHook(per_kind)` with
  `Rate(calls, per)` (GCRA bucket per kind, shared safely across `run_sync` loops), and
  `redact()`.
- Approval (`agent_adapter.approval`): `evaluate_confirm()` for `confirm` rules,
  `AutoApprove`, `AutoDeny`, `CallbackApproval`, `DeferredApproval`, the `ApprovalStore`
  protocol (atomic `create` per call, test-and-set `consume`), `ApprovalRecord` (bound to
  the requested tool and arguments by `args_digest`; a reused call id with other arguments
  fails closed), `request_digest()` and `MemoryApprovalStore` (thread-safe; first
  `decide` wins).
- `SQLiteApprovalStore` (stdlib `sqlite3`, WAL, `BEGIN IMMEDIATE`, schema v1) and
  `RedisApprovalStore` (Lua scripts, compare-and-set decide/consume, `ttl_s`): durable
  `ApprovalStore`s for approvals paused in one process and resumed in another.
- `ApprovalStoreCorrupt` (`AdapterError`, kind `internal-error`): a corrupt, tampered or
  foreign stored approval fails closed.
- Redis approval-store race tests run through a client that yields before each command,
  so a missing compare-and-set makes them fail.
- Docs: concept pages "Hooks" and "Approval".
- Cost: versioned, overridable price table `cost.Prices` (`default()`, `load()`,
  `merged()`, `resolve()` by exact name / alias / snapshot-suffix prefix, `price()`, `cost()`,
  `unit_cost()` incl. Gemini explicit-cache storage, `caps()`) with prompt-size tiers,
  dated periods, cache-write prices per TTL and caps-only entries. The packaged table
  (`cost/prices_default.json`, version `2026-10-29`) holds OpenAI and Gemini list prices;
  Anthropic is caps-only.
- `cost.Ledger` (a `LedgerPort`): prices entries on record, idempotent on
  `(run_id, entry_id)`, `max_runs` eviction, `summary()` (cost per stage and kind, prompt /
  response / tool cache hit rates, parallelism); sinks `MemorySink`, `CallbackSink`,
  `JsonlSink` and `read_jsonl()`.
- Tracing sinks: `EventLog`, `LoggingSink` (content keys only with `capture_content`),
  `CallbackEventSink`, `FanoutSink`;
  `OTelSink` (`otel` extra) exporting GenAI-convention spans and token / duration metrics,
  with every attribute and metric name in `tracing.semconv`.
- Run viewer (`agent_adapter.viewer`): `record(executor=None, folder=..., capture_content=False,
  keep_runs=200)` adds a `Recorder` that writes one JSONL file per run and process on a
  background thread (secrets masked, content only on request, old runs rotated);
  `recorder.flush()` and `recorder.stop()`. `agent-adapter ui [FOLDER]` serves a local page
  on 127.0.0.1 only, opened with a per-start token in the printed address: runs list, flow
  graph, timeline, tools and, with `--approvals DB --reviewer NAME`, Approve and Deny that
  only record the decision (`executor.resume` runs the call). `--export FILE.html [--run KEY]`
  writes one offline file.
- Chat in the run viewer: `agent-adapter ui --agent MOD:ATTR` (any agent:
  `agent(message, session)` returning text or an async iterator of chunks) or
  `--model MOD:ATTR --tools MOD:ATTR` (the built-in tool loop), with `--executor`,
  `--system`, `--max-steps`, `--turn-timeout`, `--approval-timeout`, `--reviewer` and
  `--chat-dir`; `serve(...)` does the same from Python. Public types in
  `agent_adapter.viewer`: `ChatMessage`, `ChatSession` (`context()` puts tool calls in the
  turn's run), `Attachment`, `ChatAgent`. The page streams Markdown answers (a safe subset
  built as DOM nodes), shows tool and approval cards (one-use decisions), Stop (Esc),
  edit-and-resend and Replay as branches, Compare, Save as eval case (`eval_cases.jsonl`,
  loadable by `load_cases`), a session state editor, uploads (sniffed types, 10 MB per file,
  50 MB per chat) and an inspector with Graph, Timeline, Events, State and Cost per turn.
  Sessions are saved in the chat folder. Examples: `examples/chat_minimal.py`,
  `examples/chat_builtin.py`, `examples/chat_any_agent.py`, `examples/chat_wrappers.py`;
  guides: Chat with your agent, Wrap a framework for chat.
- Chat demo in the docs: the real chat page replays recorded turns of
  `examples/chat_flows.py` (a scripted city help desk, one flow per pattern: router,
  fan-out and fan-in, approval, loop, all together; `--check` runs them offline). Suggested
  questions replace the composer; tool cards, the approval, the streamed answer and the
  inspector's live graph, timeline, events, state and cost play at a believable pace, or at
  once with reduced motion. One offline file with no server and no network. Guide: Chat
  flows by example.
- Navigable flow graph: wheel or pinch zoom at the pointer, mouse or one-finger drag to pan,
  minimap, fit, arrow-key moves along edges, `/` search, hover paths, a node panel linked to
  Timeline, Events and State, replay with a step slider, live status icons (including
  `stopped`), edge counts, Map progress, loop lanes, SVG download and Mermaid copy; layered
  layout with crossing reduction. New Events, State and Compare tabs in the Runs view.
  Guide: Navigate the graph, with the real viewer embedded on a recorded sample run.
- `FlowRunner(trace_state=False)`: `node.finished` always carries `changed` (the field names
  a node set); with `trace_state=True` also `updates` (values, JSON, 2 KB each), which the
  run recorder keeps with secret-named keys masked.
- `run_tool_loop(..., on_text=)`: streams the model's text deltas to a callback.
- `Flow.graph()` (JSON nodes and edges: fan-out, fan-in, routes, loops, pause gates) and
  `Flow.to_mermaid()`. `Executor.add_sink` / `remove_sink` and `tracing.SinkHub`.
- Cache backends: `MemoryCache` (LRU, monotonic TTLs) and `RedisCache` (extra `redis`;
  one `redis.asyncio` client per event loop, `RedisError` logged as a miss). The
  `LLMResponseCache` stores normalised responses in the JSON envelope; hits come back with
  `cached=True` and zero latency. `PromptLayout` keeps a stable, cacheable prefix.
- Provider prompt-cache plug-ins in `agent_adapter.cache.providers`, each gated by
  `ModelCaps`: `OpenAIPromptCache` (`prompt_cache_key`, `prompt_cache_retention`),
  `OpenAIPromptCacheV2` (`prompt_cache_options`, explicit breakpoints),
  `GeminiExplicitCache` (one cached content per prefix, storage recorded in the ledger),
  `GeminiImplicitCache` and `AnthropicPromptCache` (`cache_control`, at most four).
- Model ports in `agent_adapter.models`: `OpenAIModel` (Responses and Chat Completions),
  `GeminiModel`, `AnthropicModel`, `CallableModel` (wrap your own LLM function) and
  `FallbackModel` (falls over on rate limits, 5xx and timeouts; emits `llm.fallback`).
  Streaming with text and tool-call deltas, strict tools, structured output, reasoning
  effort, prompt-cache plug-in hook, one SDK client per event loop, and a typed error map
  onto `ModelError` subclasses.
- Usage normalisers (`usage_openai_chat`, `usage_openai_responses`, `usage_gemini`,
  `usage_anthropic`, `usage_voyage`, `normalise_usage`). OpenAI `input_tokens` excludes
  cached reads and cache writes; writes are reported once in `cache_write_input_tokens`,
  split by TTL.
- Integration protocol (`agent_adapter.integrations.base`): `ToolRunner`, `Integration`,
  `FlowCompiler`, `IntegrationBase`, `model_visible_error`; discovery through the
  `agent_adapter.integrations` entry-point group (`available_integrations`,
  `get_integration`), which logs and skips broken plugins; `INTEGRATION` objects for the
  OpenAI, Gemini and Anthropic renderers; Anthropic `tool_result_blocks`.
- Contract kit (`tests/integrations/kit.py`, checks C1–C8) and a sample third-party plugin
  (`tests/fixtures/demo_integration`); guide: Contributing → Writing an integration.
- `InstrumentedModel` (`agent_adapter.models`) and `Executor.model(port, cache_ttl=None)`:
  any `LLMPort` gets the executor's hooks (`before_llm` / `after_llm` / `on_llm_error`), the
  LLM response cache, one `kind="llm"` ledger entry per call (deterministic entry ids,
  `cache_write_by_ttl`, cost 0 on a cache hit) and `llm.started` / `llm.finished` /
  `llm.cached` / `llm.failed` events, for `generate` and `stream`.
- `run_tool_loop(model, registry, messages, ...)` and `ToolLoopResult`
  (`agent_adapter.exec`, also top level): a minimal tool-calling loop whose tool calls run
  through `executor.run_many`; stops on `stop`, `length`, `content_filter`, `pending` and
  `max_steps`; a history ending in an unanswered tool-call turn resumes by running it.
- OpenAI Agents SDK integration (`agent_adapter.integrations.openai_agents`, extra
  `openai-agents`): `function_tool` / `function_tools` (strict schemas, `confirm` →
  `needs_approval`), `resolve_interruptions` (port decisions → `RunState.approve` /
  `reject`; each approved call gets a one-use token that carries any edited arguments).
- Pydantic AI integration (`agent_adapter.integrations.pydantic_ai`, extra `pydantic-ai`):
  `pydantic_ai_tool` / `pydantic_ai_tools` (`Tool.from_schema`, `requires_approval` or
  `ApprovalRequired` from `confirm`, `ModelRetry` for invalid arguments) and
  `resolve_deferred` (`ToolApproved(override_args)` / `ToolDenied`).
- LlamaIndex integration (`agent_adapter.integrations.llamaindex`, extra `llamaindex`):
  `llamaindex_tool` / `llamaindex_tools` (`FunctionTool` with `fn_schema`, async `acall`
  and a `run_sync` sync path).
- CrewAI integration (`agent_adapter.integrations.crewai`, extra `crewai`, own
  environment): `crewai_tool` / `crewai_tools` (`BaseTool` with `_arun` and a `run_sync`
  `_run`; CrewAI's result cache off).
- `agent_adapter.approval.batch`: `ask_each` and the `PreparedApproval` protocol
  (`DeferredApproval.prepare` / `consume`): batches of approvals are consumed only once every
  call is decided, exactly the records that were vetted.
- `agent_adapter.flow`: the portable flow spec and its local runner. `Flow` builder
  (`node`, `edge` with parallel branches, `route` with `max_loops` / `on_exhausted`,
  `start`, `end`, `prompts` / `with_prompts`); node actions `Call`, `Map` (`tool`,
  `tool_from`, `fn`; `on_error="fail"|"skip"`), `Join`, `Route`; merge rules
  `Merge.replace` / `append` / `sum` / `keyed(by)` with `merge_rules` and `apply_updates`;
  `validate()` returns the `FlowGraph` IR and lists every problem in one
  `FlowDefinitionError`. `FlowRunner` runs LangGraph-style supersteps on the executor
  (deferred joins, node hooks, timeouts and retries, `max_steps`), pauses at `pause`
  gates or pending tool approvals with a `FlowCheckpoint`, and `resume()` re-runs only the
  paused node in the same run; `node_prompt(ctx)` / `node_model(ctx)` for node code.
  Parity cases for the LangGraph and ADK compilers in `tests/flow/parity.py`.
- LangChain: `structured_tool(s)` export registry tools as `StructuredTool`s that run through
  the executor (ToolCall id → `ctx.call_id`, model-visible errors as `ToolException`, sync
  `invoke` via `run_sync`); `import_langchain(registry, tools, prefix=, **policy)` imports
  `BaseTool`s with their model-facing schema (`InjectedToolArg` excluded).
- Gemini: `run_function_calls(calls, executor=, ctx=, registry=)` answers the model's function
  calls with function-response parts (concurrently for an `Executor`); `afc_callable(s)` give
  genai automatic function calling async proxies whose synthesised signature carries each
  property's Gemini schema.
- OpenAPI importer `import_openapi(registry, spec, base_url=, auth=, headers=, operations=,
  prefix=, timeout=, client_factory=, **policy)`: one tool per operation (flat sanitised
  schema from parameters and the JSON body, `$ref`s inlined, JSON/YAML/path documents),
  `BearerAuth` / `HeaderAuth` / `QueryAuth`, 4xx → `ToolError`, 429/5xx retried (5xx only
  for idempotent operations). Safe with untrusted documents: requests never leave
  `base_url`, path values cannot traverse, expansion bombs are bounded (an operation over a
  per-operation cap is skipped, a document over a document-wide budget is a `ConfigError`),
  malformed documents are `ConfigError`, responses capped at 10 MiB.
- Contract tests for the OpenAI, Anthropic and Gemini integrations (`contract-openai`,
  `contract-anthropic`, `contract-gemini` now run).
- MCP integration (`agent_adapter.integrations.mcp`, extra `mcp`, `mcp>=2.3` v2 API):
  `mcp_server` (lowlevel `Server` over `registry.for_target("mcp")`) and
  `streamable_http_app` (stateless Streamable HTTP, JSON responses); input schemas in the
  Anthropic-plain shape, non-object output schemas wrapped under `result`, annotations
  from `effects` / `confirm` / `idempotent` (`tool_annotations`, `output_schema_for`);
  errors as `isError` results with `ErrorInfo.message`; a pending approval answers
  `approval required: <id>` with the approval id and the resume keys in `_meta`.
  Run and call ids are scoped by the caller's principal (`principal=` hook; the HTTP app
  defaults to the bearer token and ignores anonymous client keys unless
  `allow_anonymous=True`); a client call key is honoured only to resume a pending result.
  Opt-in `client_approval=True`: the MCP 2026-07-28 `input_required` form-elicitation
  round trip with a sealed `requestState` bound to the arguments and the principal
  (`request_state_keys` for several processes). `import_mcp(registry, client, ...,
  send_ids=False)` registers remote tools with their schemas verbatim, `effects` /
  `idempotent` from the hints, remote errors as `ToolError`, and resumes remote pending
  approvals; `mcp_client(target, headers=...)` for URLs, stdio and in-process servers.
- Native-approval framework bridges, each in its own extra and module under
  `agent_adapter.integrations`: Microsoft Agent Framework (`agent_framework_tool(s)`,
  `approval_mode="always_require"` for `confirm` tools, `resolve_approvals`), Agno
  (`agno_function(s)`, `requires_confirmation`, `resolve_requirements`), Strands Agents
  (`strands_tool(s)`, `strands_approval_hook`), Haystack (`haystack_tool(s)`,
  `confirmation_hook`), Claude Agent SDK (`claude_sdk_tool(s)`, `claude_sdk_server`,
  `claude_sdk_bridge`, one per session, pairing a server with its own `can_use_tool` and
  turn-end hooks; permission-only `can_use_tool`; edits via `updated_input`) and
  Semantic Kernel (`kernel_function`,
  `kernel_plugin`, `approval_filter` for `<plugin>-<function>`). A framework's approval
  only adds a gate: a `confirm` tool skips the executor's own ask only with a one-use
  token the adapter issued after its port approved, bound to the run, the call id, the
  tool and the argument digest.
- DSPy, smolagents and AutoGen integrations (`agent_adapter.integrations.dspy`,
  `.smolagents`, `.autogen`): `dspy_tool(s)`, `smolagents_tool(s)`, `autogen_tool(s)`
  export registry tools that run through the executor (hooks, approval port, timeouts,
  retries, cache, ledger). DSPy tools also work from sync code via `run_sync`;
  smolagents' sync `forward` runs through `run_sync`, also inside a running loop; AutoGen
  tools carry the model's call id and honour the `CancellationToken`. None of the three
  has a per-tool approval gate: `confirm` tools ask the executor's `ApprovalPort`.
- HTTP tool server (`agent_adapter.server`): `ToolService` (framework-free routes for the
  catalogue, invoke, approvals, decide, run events as SSE, run cost and `/openapi.json`),
  `create_app` / `serve` (Starlette + uvicorn, MCP mounted at `/mcp` behind the same
  authenticator and `authorize`), `create_wsgi_app` (a2wsgi), `python -m
  agent_adapter.server`; RFC 9457 problems (`Problem`, `problem_for`); `Principal`,
  `bearer_auth`, `APPROVER_SCOPE`; `EventStream` / `MemoryEventStream`;
  `openapi_document`. Runs belong to the principal that started them; run, thread and call
  ids are derived from (principal, client key); approvals are visible to the invoker or an
  approver only; `Idempotency-Key` replays bind to the original request.
- `agent_adapter.server.idempotency`: `IdempotencyStore` (claim / complete / release /
  update leases), `MemoryIdempotencyStore`, `RedisIdempotencyStore`; a request with an
  `Idempotency-Key` runs once on any worker sharing the store (store errors answer 503).
  Deciding an approval re-checks `authorize` for edited arguments and for the invoker.
  REST hardening: JSON-only POSTs (415), `Host` allowlist `allowed_hosts` (421), one
  `Authorization` / `Idempotency-Key` header (400), body read timeout (408), event-stream
  caps (429); `set_log_key` for HMAC principal refs in logs; CLI `--allowed-host`,
  `--allow-anonymous-public`. The MCP server's `tools/list` honours an optional runner
  `visible(spec, ctx)` hook (the HTTP server's `/mcp` hides tools `authorize` refuses).
- `runtime.ids.scoped_id` (moved from the MCP integration; shared by MCP and HTTP).
- TypeScript client `@agent-adapter/client` (`clients/typescript/`): zero runtime
  dependencies, Node ≥ 20, ESM + CJS. `ToolClient` covers every HTTP tool-server route
  (`health`, `listTools`, `getTool`, `invoke`, `getApproval`, `listApprovals`, `decide`,
  `events`, `cost`), raises one error class per problem kind (`ToolArgsInvalidError`,
  `ApprovalDeniedError`, `BudgetExceededError`, ...), sends an `Idempotency-Key` on every
  invoke and decide (stable across its retries), retries only requests that are safe to
  repeat, and parses server-sent events with a bounded buffer and `Last-Event-ID` reconnect.
  `toolDefinitions` and `asVercelTools` adapt the catalogue to TypeScript agent frameworks.
- Optimiser (`agent_adapter.optimiser`): `optimise()` runs Propose → Probe → Compose over a
  host's eval cases offline and writes `harness.patch`, `report.md`, `result.json`,
  `runs/<label>/<case>.md` and `harness.after.json`, never applying anything. `RegistryHost`
  (tool loop over a registry) and `FlowHost` (a flow's node prompts) cover the common hosts;
  `apply_tree` applies a reviewed tree. Budget identity `(1 + 3k) × N` case runs and `2k`
  proposer calls; `OptimiserBudget` limits stop early with a partial report; finished stages
  resume from `state.json`; lessons go to `lessons.json`. CLI:
  `python -m agent_adapter.optimiser --host mod:fn --model provider:model [--dry-run]`
  (exit 0 chosen, 1 error, 2 KEEP_BASE, 3 stopped by a budget limit before a choice).
- Google ADK 2.x (`agent_adapter.integrations.adk`): `adk_tool` / `adk_tools` export each
  spec as a `BaseTool` (Gemini schema family) whose `check_require_confirmation` applies the
  `confirm` rule; `resolve_confirmations(events, port, registry)` answers ADK's tool
  confirmations from the approval port (`ask_each`) with one-use tokens bound to invocation
  id, call id, tool and argument digest (edits supported). ADK's `confirmed` alone never
  approves. `compile_flow(flow)` builds a `Workflow` with one superstep driver node whose
  gates, node visits and answers are child nodes with deterministic run ids, so ADK replays
  them on resume instead of running them again; `run_workflow` and `resume_message` run and
  resume it. Every parity case matches the local runner.
- LangGraph integration (`agent_adapter.integrations.langgraph`, extra `langgraph`):
  `tool_node(registry, executor=, messages_key=, concurrency=)` runs an `AIMessage`'s tool
  calls through the executor (confirm calls one at a time after the others),
  `tool_task(spec, into=)` + `fan_out(target, over)` for `Send` fan-outs, `reducer(rule)`
  channels for the flow merge rules, and `LangGraphApproval()`, an approval port that
  pauses the graph with `interrupt()`. Every resume value must carry the interrupt's
  binding (run, tool, argument digest, call id) and an HMAC under the port's key
  (`LangGraphApproval(key=).resume_value(...)`); anything else is refused as an invalid
  resume, and each answer is used once per task run. With a store-backed port only the
  store decides. `compile_flow(flow, executor=, checkpoint_key=)` compiles a `Flow` to a
  `StateGraph` (gates as `interrupt_before`, tool approvals as pauses), and
  `run_compiled(app, state, config)` runs it to a `FlowResult`; every flow parity case
  matches the local runner, also when a run resumes in another process (the paused node's
  progress is sealed into its interrupt value).
- `agent_adapter.embeddings`: `EmbeddingPort`, `EmbeddingResult`, `CallableEmbedder` (wrap any
  sync or async embedding function; batching, `timeout_s`, per-text `EmbeddingCache`, one
  `kind="api"`, `stage="embed"` ledger entry per call) and `embed_key`;
  `embedding.started` / `.finished` / `.failed` trace events and OTel `embeddings {model}`
  spans and metrics (never the texts); `runtime.current.resolve_context`.
- Cost: prices for your own models. Pattern keys (`"acme-*"`, catch-all `"*"`; exact names
  and aliases still win), `free: true` entries (cost 0, no warning), `per_call` and
  `per_second` prices (the ledger uses `latency_ms`), and `Prices.empty()`.
  `Prices.cost()` takes `seconds=` and `attempts=`; `ResolvedPrice` gains `per_call`,
  `per_second`, `free`. Unit prices take the same patterns and `free: true`; `per_call` is
  charged per attempt. The `llm.finished` event cost matches the ledger entry.
- How-to "Bring your own model": wrap any local or remote model with `CallableModel`, or
  point `OpenAIModel` at Ollama, vLLM, LM Studio or llama.cpp server via `client_kwargs`.
- `agent_adapter.errors`: every error class and `ErrorInfo` in one module.
- Public homes for the ports host code implements: `LLMPort`, `PromptCachePlugin` and
  `ModelCaps` in `agent_adapter.models`, `CachePort` in `.cache`, `LedgerPort` in `.cost`,
  `EventSink` in `.tracing` (`ApprovalPort` stays in `.approval`); `is_invoker` in
  `agent_adapter.server`.
- `openapi_document(dialect="3.0" | "3.0-gateway")` and `GET /openapi-3.0.json`: OpenAPI 3.0.3
  without `oneOf`/`anyOf`/`allOf` (nullable flags, flattened unions) for Go and
  Rust generators and AgentCore Gateway OpenAPI targets; `/readyz` and the document routes
  are now in the OpenAPI document.
- MCP tasks in a shared store: `TaskOptions(store=RedisTaskStore(...))` (also
  `MemoryTaskStore`, `TaskStore`), so any worker answers `tasks/*`; a task whose worker died
  reads as failed.
- `reset_default_executor()` in `agent_adapter.exec`: drops the default executor, so the next
  use makes a fresh one (for tests).
- **Context compression** (`agent_adapter.context`): `CompressionPolicy` runs its steps in
  order until the history is under target, with your own models, embedders and rerankers.
  Turn it on per call (`LLMRequest(compression=policy)`, honoured by every built-in port),
  for every call of a model (`CompressingModel(model, policy, when=None)`) or in the agent
  loop (`run_tool_loop(..., compression=policy)`: compressed before each call, and the loop
  goes on with the shorter history).
- Free steps (no model call): `RemoveOldMessages`, `RemoveDuplicates` (exact),
  `CutLongMessages`, `ShrinkToolResults` (with `recall_tool` to read a saved result back) and
  `MyOwnStep`. Tool rounds stay whole; pinned and approval messages stay verbatim; a system
  prompt is shortened only with `include_system_prompt=True`, never removed.
- AI steps: `Summarize`, `SummarizeLongMessage`, `KeepMostRelevant` and
  `RemoveDuplicates(match="similar")`, with the scorers `EmbeddingScorer`, `RerankerScorer`
  and `MyScorer`, and `RerankerPort` / `CallableReranker` (`RerankResult`). Summary calls are
  `llm` ledger entries and rerank calls priced `api` entries (stage `compress`); new
  `rerank.started`, `rerank.finished` and `rerank.failed` events; `default_steps(summary_model)`.
- `compress_for(messages, policy)` and `compress_for_sync`: any framework's own message list
  back in its own type, kept messages as the very same objects (`agent_adapter.context.formats`).
  Formats `agent_adapter`, `langchain`, `openai` (chat and Responses), `anthropic`, `gemini`,
  plus `pydantic_ai`, `agno`, `strands`, `smolagents`, `autogen`, `ag2`, `llamaindex`,
  `semantic_kernel`, `agent_framework` and `camel`, registered when their integration is
  imported; CrewAI and DSPy use `openai`. `format="auto"`, `pinned=`, and `register_format`
  for your own format.
- Framework hooks: `langgraph_compressor(policy)`, `pydantic_ai_history_processor`,
  `agno_compressed_model`, `strands_compress_middleware`, `smolagents_compressed_model`,
  `autogen_model_context` and `ag2_compress_policy`; tested snippets for every other
  framework on the Compression by framework page.
- `ModelContextOverflow` (a `ModelBadRequest`, kind `model-context-overflow`), mapped from
  the OpenAI, Anthropic and Gemini ports; a policy's `if_still_too_long` runs once and the
  call is retried once.
- `CompressionPolicy(on_step_error="skip" | "raise")`: by default a failing step is logged,
  recorded as a `StepFailure` in `CompressionReport.steps_failed` and skipped, so the model
  call goes on (`ConfigError` always raises).
- `Message.pinned`: a pinned message is never removed or changed (dumped only when true).
- `context.compressed` and `context.compress_failed` events (counts only, never text) and
  a `compress_context` OTel span.
- `ModelCaps.context_window`: a percent policy without `model_max_tokens` reads it; a clear
  `ConfigError` when no window is known.
- Docs: Context compression, Compress a long context, Compression by framework.
- `cache_for_seconds` (summaries) and `save_for_seconds` (saved tool results) lifetimes.

### Changed

- `Executor.sink` is a stable `SinkHub` and never `None`: the sink you passed, then every
  sink added later. Contexts and models made from the executor hold the hub, so a sink
  added later reaches them too. To check what it sends to, read `executor.sink.sinks`
  (`executor.sink is my_sink` and `executor.sink is None` no longer hold). A hub with no
  sinks counts as no sink, so events still go to the running executor's sinks.
- `FlowRunner(sink=...)` adds its sink to the run's sinks instead of replacing them, and a
  sink that is already there gets each event once.
- `configure` and `set_default_executor` move the sinks `record()` attached to the default
  executor onto the new default; `reset_default_executor` detaches and forgets them.
- `Executor.registry` is never `None`: `Executor()` resolves names in the default registry
  (it used to raise `ToolNotFound`).
- The top level is a curated, lazy front door of 55 names (`agent_adapter.__all__`);
  `import agent_adapter` loads no other module and no pydantic. Every other error and
  `ErrorInfo` moved to `agent_adapter.errors`, and `ShortCircuit` to `agent_adapter.hooks`;
  a moved name raises an `AttributeError` / `ImportError` that names its new home.
- `CallableModel` runs a sync `fn` in a worker thread (context variables copied), so a
  blocking local model or HTTP client never stalls the event loop. A sync `fn` that returns
  an awaitable is awaited.
- `CallableModel` rejects a `usage` mapping with keys that are not `Usage` fields
  (`ModelOutputInvalid`), instead of silently billing $0.

- Version 1.0.0.
- `all` extra: final membership, verified with `pip check` and every contract test in the
  `contract-all-compatible` session. Added `llamaindex` and `claude-agent`. `dspy` is not
  in `all`: it pulls in LiteLLM.
  Not in `all` (own environment): `crewai` (needs `mcp<2`), `strands` (needs `mcp<2.2`),
  `semantic-kernel` (pins `pydantic<2.14`), `adk` (pins `opentelemetry<=1.44`).
- Core floor lowered to `pydantic>=2.12,<3` (suite verified on 2.12.5; floor of mcp,
  openai-agents, pydantic-ai and google-adk; unlocks `crewai>=1.15`).
- `gemini` extra floor raised to `google-genai>=2.30`.
- `langgraph` extra floor raised to `langgraph>=1.2.15` (`interrupt(..., response_schema=)`).
- `__version__` now lives in `agent_adapter/_version.py` (still re-exported).
- Prompt-cache plug-ins without `caps=` now read `Prices.default().caps`: known models get
  their cache parameters, unknown models still get none. Pass `caps=lambda m: ModelCaps()`
  for the old all-off behaviour.
- `FallbackModel`: with `metadata["call_id"]` set, every hop `i` (hop 0 too) receives
  `{call_id}:hop{i}`, so instrumented hops record separate ledger entries.
- `OpenAIModel` / `AnthropicModel` fill the `platform` caches once per process on Windows
  without WMI (cpython#125315 process crash).
- The demo integration fixture defaults `executor=None` to `exec.defaults.runner_for`.
- The `claude-agent` extra also installs `mcp` (the bridge runs its own MCP server).
- Tools from `spec_from_schema` (MCP, LangChain and OpenAPI imports) build their pydantic
  argument model on first use, then keep it; behaviour is unchanged.
- OpenAPI import: the input-fields budget is capped at 100 000 and checked before an
  operation holds any schema, so skipped operations never exhaust a document-wide budget;
  an operation named in `operations=` that a bound skips is a `ConfigError`.
- `Executor.as_completed` is typed `AsyncGenerator[ToolResult, None]`, so
  `contextlib.aclosing(...)` type-checks.
- Tools exported without an executor read the default executor at each call, so
  `set_default_executor` after export applies; `compile_flow` still reads it once.
- Every integration module declares `__all__`, so imported helpers no longer look public.
- `Executor.run_plan` takes `registry=`, like the other batch methods.
- `MemoryApprovalStore.decide` refuses `edited_args` that are not plain JSON
  (`ToolUsageError`), as the SQLite and Redis stores do.
- The `server` extra installs `mcp`, because `create_app` mounts `/mcp` by default.

- The `server` extra caps `h11` at `<1`.
- The Claude Agent SDK version gate matches the `claude-agent` extra at patch level
  (`0.2.150` to below `0.3.0`).

- Model-call ledger entries arrive already priced (one measured duration, shared with the
  `llm.finished` event), so a custom `LedgerPort` with `prices` no longer sees `cost=None`.
- `per_second` prices bill only the time inside the model port, never hooks, cache lookups
  or a slow stream consumer.
- `Executor.resume` needs an explicit decision and `decided_by` (the argument or the
  decision's own): a pending record without them raises `ToolUsageError`.
- Docs: the nav has 10 tabs instead of 20; `llms.txt` is generated from the nav.
- Secret redaction judges a key by its words (`is_secret_name`, and the new
  `is_secret_field` and `redact_secrets`): plural names (`api_keys`, `passwords`) and a bare
  `key` holding a secret-like value are masked, while `max_tokens` or `token_count` stay
  visible; `LoggingHook` applies it on top of its patterns, and compression redacts only
  secret-named or secret-shaped values before a summary call.
- Summaries are cached only in a cache given to the step or the policy (`cache=`), never in
  `ToolContext.cache` by default.

### Removed

- `openai_agents.ApprovalOverrides` and the `overrides=` keyword of `function_tool`,
  `function_tools` and `resolve_interruptions`: edits now travel in the one-use token.
- The unused `optimiser._fatal.FATAL_KINDS`.

### Fixed

- Optimiser: a host-reported `metrics["cost_usd"]` now counts toward `max_usd` when the
  ledger saw no spend for that case run (external agents bypassed the budget).
- A deeply nested checkpoint is refused as `checkpoint-tampered` instead of raising
  `RecursionError`.
- `create_wsgi_app` returns a `WsgiApp` with `close()` (or a `with` block) that ends its
  lifespan and loop thread, so the lifespan task is no longer leaked.
- `WsgiApp.close()` cancels request tasks before the lifespan ends: one whose response has not
  started gets a 503 problem, one mid-response is aborted, so no worker thread waits forever
  and a response closed unread no longer leaves "Task was destroyed but it is pending!".
  `close()` waits at most `CLOSE_TIMEOUT_S` for open responses; a request after `close()`
  gets a 503 `service-unavailable` problem instead of "Event loop is closed".
- `agent_adapter.embeddings` loads on attribute access (`agent_adapter.embeddings`), like
  every other subpackage.

- `integrations.langgraph.one_use` raises `MissingDependency` naming `agent-adapter[langgraph]`
  instead of a raw `ModuleNotFoundError` when langgraph is not installed.
- `streamable_http_app` and `mcp_server` check for `mcp` first, so a missing `mcp` raises
  `MissingDependency` before any misleading auth warning.

- Embeddings: vectors must be real numbers (no str, bool, inf or NaN), a batch is checked
  before it is cached, and the cache key cannot collide when provider or model contain `:`.
- OpenAPI 3.1: tagged-union `discriminator.mapping` pointed into `$defs`; it now points into
  `components/schemas`.
- MCP tasks: a failed task logged its error text at WARNING; it is logged by type (text at DEBUG).
- A redelivered queue resume of a consumed approval is acked as `already_done`, not sent to
  the DLQ.
- DynamoDB store: throttling is no longer retried as a lost race, and retries use jittered
  backoff; the SQLAlchemy store refuses a table without its partial unique index.
- OpenAPI 3.0 gateway document: a flattened union keeps its tag as an `enum` and no longer
  carries a dangling `discriminator`.
- Optimiser JSONL cases split on newlines only, so a U+2028 inside a value no longer splits
  a line.
- `compress_for` no longer rebuilds a message's original native when a step changed its
  role or ids, which gave orphan tool results and a second system prompt.
- `run_tool_loop` seeds its ledger entry numbers from the ledger as well as the history, so
  a resumed or compressed history no longer reuses paid entry ids (the ledger dropped the
  new entries as duplicates).

### Security

- Framework approvals: a framework's approval signal only adds a gate; a `confirm` tool
  runs only on the executor's own port decision or a one-use token the adapter issued,
  bound to the run, call id, tool and argument digest.
- HTTP server security model: principal-scoped run, thread and call ids, owner-only runs,
  approvals visible to the invoker or an approver, fail-closed hooks. Residual risks are listed under "Security model" on the
  HTTP server page.
- OpenAI Agents / Pydantic AI bridges: a framework's own approval (`RunState.approve`,
  `always_approve`, stored run state, `tool_call_approved`, hand-built
  `DeferredToolResults`) never approves a `confirm` tool; only a one-use token from
  `resolve_interruptions` / `resolve_deferred` does. The executor always runs the
  `confirm` rule itself.
- HTTP server: a decision on a record whose tool the server does not register takes no
  edited arguments (400) and approves only when the record names an invoker (403).
- MCP `streamable_http_app`: a non-local `host` without auth is a `ConfigError` unless
  `allow_anonymous=True`; `auth=` and `token_verifier=` must be passed together.
- HTTP server: `create_app` / `serve` refuse an anonymous app on a non-local bind host
  (`ConfigError`) unless `allow_anonymous_public=True` (CLI `--allow-anonymous-public`).
- An anonymous HTTP app, or an MCP app on a local host without auth, serves loopback
  clients only: any other client gets a 403 `non-local-client`.
- Edited arguments from an approval re-run the `before_tool` hooks, so guardrails and
  budgets check what actually runs; `RateLimitHook` books a second slot for the edit.
- A pre-approval on the context (`ctx.approval`) covers one call of `run` / `run_spec` /
  `run_call`; `run_many`, `as_completed`, `run_plan` and `run_tool_loop` never use it, and
  the tool body and its nested or loop calls never inherit it.
- MCP: a custom `principal=` is not auth. It follows the no-auth rules, and a request it
  names no caller for (`None`) is refused unless `allow_anonymous=True`.
- Pydantic AI: a token is bound to the paused call's `ToolCallPart`, so another run or a
  reloaded history gets none; one resolve issues at most 256 tokens on either bridge, and
  the rest log a `WARNING` and ask the executor's port.
- Docs: never put an anonymous app behind a reverse proxy or a unix socket that remote
  clients reach; the loopback guard cannot see past it.
- Approval stores: `RedisApprovalStore` needs a `prefix` ending in `:`, new SQLite files
  are created `0600`, and the Approval stores page covers access and permissions.
- Approval stores log a `WARNING` when eviction drops a decided, unused approval; the docs
  ask for Redis `maxmemory-policy noeviction` for every `ttl_s`.
- Exception text and tracebacks are no longer logged above `DEBUG`: failures are logged
  by exception type at `WARNING` / `ERROR`; guardrail reasons are logged only at `DEBUG`.
- An AST test (`tests/test_log_hygiene_scan.py`) scans the source so no `WARNING`+ log call
  carries exception text or a traceback.
- HTTP server: what happens when the idempotency store fails after a run (the response
  stands, a `WARNING` is logged) is documented in the Security model.
- Dependency floors: `server` extra `h11>=0.16` (CVE-2025-43859); `claude-agent` extra
  `mcp>=2.3,<3`, the same floor as the `mcp` extra.
- MCP docs: `client_approval=True` makes the remote MCP client the approver, and an
  agentic host may accept the form without a person.
- TypeScript client: redirects are never followed (`redirect: "error"`), so the bearer
  token never goes to another URL.
- Docs run model tool calls through the executor, never `spec.invoke`; a test fails on
  `.invoke(` in the newcomer and architecture pages and their examples.
- Install docs point pip at the internal index (never `--extra-index-url`), against
  dependency confusion.

- The sdist is allow-listed (`only-include`): `agent_adapter/`, `README.md` and `CHANGELOG.md`
  only, so `node_modules` or an `.env` file in the tree can never ship.
- The `Private :: Do Not Upload` classifier makes public PyPI refuse an accidental upload.
- A2A: outbound connects are pinned to the IP the SSRF guard checked (no DNS rebinding),
  and the default task store is bounded (`bounded_task_store`: TTL, total and per-owner caps).
- Approvals parked by the HTTP tool server are refused outside the server flow on every path
  (`resume`, `run_call`, queue envelopes).
- AG-UI: `agui_endpoint(allowed_hosts=)` answers 421 to other `Host` headers (an anonymous
  endpoint serves loopback names only by default), and interrupts are capped per principal.
- Context compression: approval items in every message format (MCP approval requests and
  responses, ADK confirmations) are never dropped or changed; a summary is a quoted, escaped
  record marked "reference only, not instructions", and lines that read as orders are
  removed (a mostly-orders summary is refused); a system prompt is summarised only inside
  `<reference>` blocks; saved tool results are scoped to their thread or run; secrets are
  redacted before a summary call; the FastAPI example accepts only user and assistant text.
- MCP `oauth_client` checks `redirect_uri` (RFC 8252); `agent-adapter init` refuses symlinks
  and paths outside the target folder.

### Notes

- Checked offline only; verify with live keys: the OpenAI usage formula
  (`input = prompt − cached − cache_write`), whether Anthropic structured output needs
  `additionalProperties: false`, and the snake_case keys `google-genai` sends to Gemini.
- The TypeScript client (`clients/typescript`) stays `private: true` until a
  registry is chosen (see the Roadmap).

## [0.1.0] - 2026-10-29

### Added

- `@registry.tool` and `registry.wrap()` build a `ToolSpec` from a function's signature
  and Google-style docstring; `Annotated`/`Field()` descriptions and constraints are kept,
  and beat the docstring. `Field(...)` works as a parameter default.
- Tools from bound methods (`registry.wrap(obj.method)`), callable objects and
  `functools.partial`; a partial's keyword bindings act as `fixed=` values (hidden,
  never overridable).
- Parameter shaping: `fixed=`, `hide=` (a list or one name), `rename=`, `result=`,
  `name=`, `description=`; `ToolSpec.llm_to_python` holds the rename map.
- Build-time checks: LLM-facing argument names (portable keys, no leading `_`, no pydantic
  `BaseModel` attribute, no rename collisions), conflicting options, one `ToolContext`
  parameter at most, no parameter aliases (use `rename=`); `*args`, `**kwargs`,
  positional-only and unannotated parameters are rejected.
- Execution-policy options stored on the spec and type-checked at build time: `kind`,
  `tags`, `parallel_safe`, `once_per_run`, `confirm`, `timeout`, `retries`, `cache_ttl`,
  `prompt_hint`, `portable`. Empty tags and non-finite timeouts are rejected.
- `ToolContext` injection (subclasses and `Optional` forms included), hidden from the schema.
- `ToolSpec.validate_args()`, `call()` and `invoke()`; sync tools run in a worker thread.
  `ToolSpec` compares and hashes by identity; `json_schema()` is cached and returns a copy.
- `ToolSpec.returns` and `result_schema()` from the return annotation.
- Strict-mode `null` means "use the default" for every optional argument whose default is
  not `None`, at any depth and through aliases.
- `Registry`: `add`, `get`, `remove`, `subset(names=, tags=, kinds=)` (filters combine with
  AND), `catalogue()`.
- Portable-schema check at registration (`SchemaNotPortable`), with `portable=False` to opt
  out; rejects multi-item `allOf` and `type` lists other than `["X", "null"]`.
- Renderers: OpenAI Chat Completions and Responses (strict by default), Gemini function
  declarations and `genai_tool()`, Anthropic Messages tools. OpenAI strict and Gemini
  write each argument's default into its description (`Default: 8.`).
- Errors under one base, `AdapterError`: `ToolDefinitionError`, `DuplicateTool`,
  `SchemaNotPortable`, `ToolNotFound`, `ToolArgsInvalid` (JSON-safe `errors`, no input
  values), `ToolUsageError` (host misuse, also a `TypeError`), `MissingDependency` (also an
  `ImportError`).
- Documentation site (`mkdocs build --strict`) and `scripts/check.py` running every gate.

### Notes

- Requires `pydantic>=2.13,<3`: the rendered schemas are pinned by snapshot tests against
  pydantic 2.13+ output.
- The `gemini` extra requires `google-genai>=1.21`, the first release whose
  `FunctionDeclaration` has `parameters_json_schema`.
