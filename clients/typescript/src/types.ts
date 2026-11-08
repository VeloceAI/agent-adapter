/** Wire payloads of the agent-adapter HTTP tool server (contract §15.3) and client options (§16). */

/** Catalogue shape: neutral descriptors, or with a provider-specific `provider` field. */
export type Shape = "neutral" | "openai" | "anthropic" | "gemini";

/** JSON Schema object as served by the catalogue. */
export type JsonSchema = Record<string, unknown>;

export interface ToolDescriptor {
  name: string;
  description: string;
  input_schema: JsonSchema;
  output_schema: JsonSchema | null;
  kind: string;
  tags: string[];
  effects: string | null;
  idempotent: boolean | null;
  approval: "never" | "always" | "conditional";
  requires_approval: boolean;
  parallel_safe: boolean;
  /** Seconds. */
  timeout: number | null;
  retries: number;
  /** Seconds. */
  cache_ttl: number | null;
  prompt_hint: string | null;
  /** OpenAI chat tool, Anthropic tool or Gemini declaration; only when `shape` is not neutral. */
  provider?: Record<string, unknown>;
}

export interface Catalogue {
  version: string;
  tools: ToolDescriptor[];
}

export type InvokeOk<T = unknown> = {
  status: "ok";
  run_id: string;
  call_id: string;
  result: T;
  cached: boolean;
  attempts: number;
};

export type InvokePending = {
  status: "pending_approval";
  run_id: string;
  call_id: string;
  approval_id: string;
};

export type InvokeResult<T = unknown> = InvokeOk<T> | InvokePending;

export type Denied = {
  status: "denied";
  run_id: string;
  call_id: string;
  approval_id: string;
  reason: string;
};

export type DecideResult<T = unknown> = Extract<InvokeResult<T>, { status: "ok" }> | Denied;

export interface ApprovalDecisionRecord {
  approved: boolean;
  edited_args: Record<string, unknown> | null;
  reason: string;
  decided_by: string;
  /** Epoch seconds. */
  decided_at: number;
}

export interface Approval {
  approval_id: string;
  status: "pending" | "approved" | "denied";
  tool: string;
  arguments: Record<string, unknown>;
  run_id: string;
  call_id: string;
  hint: string;
  /** Epoch seconds. */
  created_at: number;
  decision: ApprovalDecisionRecord | null;
  /** Similar past decisions, sent to approvers only. Advice: the call still waits. */
  context?: ApprovalContext;
}

/** Counts of earlier decisions on similar calls; never argument values or secret fields. */
export interface ApprovalContext {
  tool: string;
  similar: number;
  approved: number;
  denied: number;
  deciders: string[];
  /** Epoch seconds. */
  last_decided_at: number | null;
  examples: Record<string, unknown>[];
  summary: string;
}

export interface StageSummary {
  calls: number;
  cost_usd: number;
  input_tokens: number;
  cached_input_tokens: number;
  cache_write_input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  errors: number;
}

export interface LedgerSummary {
  run_id: string | null;
  total_usd: number;
  unknown_cost_entries: number;
  by_stage: Record<string, StageSummary>;
  by_kind: Record<string, StageSummary>;
  prompt_cache_hit_rate: number;
  response_cache_hit_rate: number;
  tool_cache_hit_rate: number;
  wall_ms: number;
  busy_ms: number;
  parallelism: number;
}

/** One budget from `GET /budget`: limit, settled spend, live holds and its UTC window. */
export interface Budget {
  name: string;
  scope: "principal" | "user" | "team" | "global";
  /** The user id, team id, or "*" for the global budget. */
  key: string;
  /** "all", "day:YYYY-MM-DD" or "month:YYYY-MM". */
  window: string;
  limit_usd: number;
  spent_usd: number;
  reserved_usd: number;
  remaining_usd: number;
  /** Epoch seconds when the window resets; null when it never does. */
  resets_at: number | null;
  on_exceed: "stop" | "warn";
}

/** RFC 9457 problem details; extensions (`tool`, `errors`, `approval_id`, ...) are kept. */
export interface Problem {
  type: string;
  title: string;
  status: number;
  detail?: string;
  [extension: string]: unknown;
}

/** One server-sent event; `data` is the parsed JSON (an `AdapterEvent` for run events). */
export interface RunEvent {
  id?: string;
  event: string;
  data: unknown;
}

/** `Event.to_dict()` of the Python package: the `data` of a run event. */
export interface AdapterEvent {
  type: string;
  run_id: string;
  call_id: string;
  parent_id: string;
  /** Epoch seconds. */
  ts: number;
  data: Record<string, unknown>;
}

/** Retry policy for requests that are safe to repeat; `false` disables retries. */
export interface RetryOptions {
  /** Extra attempts after the first; default 2. */
  maxRetries?: number | undefined;
  /** First backoff delay; doubles per attempt with full jitter; default 250. */
  baseDelayMs?: number | undefined;
  /** Longest wait, also the largest `Retry-After` honoured; default 10_000. */
  maxDelayMs?: number | undefined;
}

export interface ClientOptions {
  /** Server root including any prefix, e.g. `https://tools.example/api`. */
  baseUrl: string;
  /** Bearer token, or a function called before every attempt (token refresh). */
  token?: string | (() => string | Promise<string>) | undefined;
  fetch?: typeof fetch | undefined;
  /** Per attempt, until the response body is read (for `events`: until headers); default 30_000. */
  timeoutMs?: number | undefined;
  headers?: Record<string, string> | undefined;
  retry?: RetryOptions | false | undefined;
  /** Longest SSE line or event payload, in characters; default 1_048_576. */
  maxEventSize?: number | undefined;
  /** An event stream silent this long counts as dropped and reconnects; default 45_000 (3 keep-alives). */
  idleTimeoutMs?: number | undefined;
  /** Largest JSON or error body read, in bytes; default 16 MiB. */
  maxResponseSize?: number | undefined;
}

export interface InvokeOptions {
  runId?: string | undefined;
  threadId?: string | undefined;
  /** Sent as `Idempotency-Key`; one is generated when absent and reused across retries. */
  idempotencyKey?: string | undefined;
  signal?: AbortSignal | undefined;
}

export interface DecideInput {
  approved: boolean;
  editedArguments?: Record<string, unknown> | undefined;
  reason?: string | undefined;
}

export interface DecideOptions {
  /** Sent as `Idempotency-Key`; one is generated when absent and reused across retries. */
  idempotencyKey?: string | undefined;
  signal?: AbortSignal | undefined;
}

export interface EventsOptions {
  signal?: AbortSignal | undefined;
  /** Resume after this event id (`Last-Event-ID`). */
  lastEventId?: string | undefined;
  /** Event types after which the server ends the stream. */
  closeOn?: string[] | undefined;
}

export interface ListToolsOptions {
  shape?: Shape | undefined;
  tags?: string[] | undefined;
  kinds?: string[] | undefined;
  signal?: AbortSignal | undefined;
}

export interface ToolDefinitionsOptions {
  tags?: string[] | undefined;
  /** Groups every call made through the definitions into one run (approvals, cost, events). */
  runId?: string | undefined;
  threadId?: string | undefined;
}

/** Second argument of `execute`, as passed by the Vercel AI SDK and similar hosts. */
export interface ExecuteOptions {
  abortSignal?: AbortSignal | undefined;
}

export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: JsonSchema;
  /** Returns `result` when the call ran, `{ pending_approval: approval_id }` when it waits. */
  execute: (input: Record<string, unknown>, options?: ExecuteOptions) => Promise<unknown>;
}

/** The part of `ToolClient` the adapters need. */
export interface ToolInvoker {
  listTools(options?: ListToolsOptions): Promise<ToolDescriptor[]>;
  invoke<T = unknown>(name: string, args: Record<string, unknown>, options?: InvokeOptions): Promise<InvokeResult<T>>;
}
