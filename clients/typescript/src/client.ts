/** `ToolClient`: every route of the agent-adapter HTTP tool server (contract §15.2, §16). */
import { asVercelTools, toolDefinitions } from "./adapters.ts";
import { EventStreamError, ToolProtocolError, ToolTransportError } from "./errors.ts";
import { checkNumber, HttpClient, MAX_TIMER_MS, newIdempotencyKey, sleep } from "./http.ts";
import type { HttpResponse } from "./http.ts";
import { YieldedEvents } from "./replay.ts";
import { DEFAULT_MAX_EVENT_SIZE, readEventStream, SseParser } from "./sse.ts";
import type { SseMessage } from "./sse.ts";
import type {
  Approval,
  ClientOptions,
  DecideInput,
  DecideOptions,
  DecideResult,
  EventsOptions,
  InvokeOptions,
  InvokeResult,
  LedgerSummary,
  ListToolsOptions,
  RunEvent,
  Shape,
  ToolDefinition,
  ToolDefinitionsOptions,
  ToolDescriptor,
  ToolInvoker,
} from "./types.ts";

type SignalOption = { signal?: AbortSignal | undefined };

/** Three of the server's 15 s keep-alives. */
const DEFAULT_IDLE_TIMEOUT_MS = 45_000;

/** Client for one tool server; safe to share across concurrent calls. */
export class ToolClient implements ToolInvoker {
  readonly #http: HttpClient;
  readonly #maxEventSize: number;
  readonly #idleTimeoutMs: number;

  constructor(options: ClientOptions) {
    this.#http = new HttpClient(options);
    this.#maxEventSize = new SseParser({ maxSize: options.maxEventSize ?? DEFAULT_MAX_EVENT_SIZE }).maxSize;
    this.#idleTimeoutMs = checkNumber(
      "idleTimeoutMs",
      options.idleTimeoutMs ?? DEFAULT_IDLE_TIMEOUT_MS,
      1,
      MAX_TIMER_MS,
    );
  }

  /** `GET /healthz`: liveness and the server's package version (no auth needed). */
  async health(options: SignalOption = {}): Promise<{ status: string; version: string }> {
    const res = await this.#http.request<{ status: string; version: string }>({
      method: "GET",
      path: "/healthz",
      signal: options.signal,
    });
    return expectRecord(res, "GET /healthz");
  }

  /** `GET /tools`: the catalogue, optionally in a provider shape and filtered by tags or kinds. */
  async listTools(options: ListToolsOptions = {}): Promise<ToolDescriptor[]> {
    const res = await this.#http.request<unknown>({
      method: "GET",
      path: "/tools",
      query: { shape: options.shape, tags: joinList(options.tags), kinds: joinList(options.kinds) },
      signal: options.signal,
    });
    const body = expectRecord(res, "GET /tools");
    if (!Array.isArray(body["tools"])) throw new ToolProtocolError("GET /tools: no tools list", res.status);
    return body["tools"] as ToolDescriptor[];
  }

  /** `GET /tools/{name}`: one descriptor. */
  async getTool(
    name: string,
    options: { shape?: Shape | undefined; signal?: AbortSignal | undefined } = {},
  ): Promise<ToolDescriptor> {
    requireText("name", name);
    const res = await this.#http.request<ToolDescriptor>({
      method: "GET",
      path: `/tools/${encodeURIComponent(name)}`,
      query: { shape: options.shape },
      signal: options.signal,
    });
    return expectRecord(res, "GET /tools/{name}");
  }

  /** `POST /tools/{name}/invoke`: runs the tool, or returns `pending_approval` when it needs a reviewer. */
  async invoke<T = unknown>(
    name: string,
    args: Record<string, unknown>,
    options: InvokeOptions = {},
  ): Promise<InvokeResult<T>> {
    requireText("name", name);
    if (!isRecord(args)) throw new TypeError("args must be a JSON object");
    const key = idempotencyKey(options.idempotencyKey);
    const body: Record<string, unknown> = { arguments: args };
    if (options.runId !== undefined) body["run_id"] = options.runId;
    if (options.threadId !== undefined) body["thread_id"] = options.threadId;
    const what = "POST /tools/{name}/invoke";
    const res = await this.#http.request<unknown>({
      method: "POST",
      path: `/tools/${encodeURIComponent(name)}/invoke`,
      body,
      headers: { "idempotency-key": key },
      signal: options.signal,
      idempotent: true,
    });
    const result = expectRecord(res, what);
    if (res.status === 200 && result["status"] === "ok") return result as InvokeResult<T>;
    if (res.status === 202 && result["status"] === "pending_approval" && typeof result["approval_id"] === "string") {
      return result as InvokeResult<T>;
    }
    throw new ToolProtocolError(
      `${what}: unexpected ${res.status} body status ${String(result["status"])}`,
      res.status,
    );
  }

  /** `GET /approvals/{id}`. */
  async getApproval(id: string, options: SignalOption = {}): Promise<Approval> {
    requireText("id", id);
    const res = await this.#http.request<Approval>({
      method: "GET",
      path: `/approvals/${encodeURIComponent(id)}`,
      signal: options.signal,
    });
    return expectRecord(res, "GET /approvals/{id}");
  }

  /** `GET /approvals`: the approvals of runs the principal owns (all of them for reviewers). */
  async listApprovals(
    options: {
      runId?: string | undefined;
      status?: Approval["status"] | undefined;
      signal?: AbortSignal | undefined;
    } = {},
  ): Promise<Approval[]> {
    const res = await this.#http.request<unknown>({
      method: "GET",
      path: "/approvals",
      query: { run_id: options.runId, status: options.status },
      signal: options.signal,
    });
    const body = expectRecord(res, "GET /approvals");
    if (!Array.isArray(body["approvals"])) throw new ToolProtocolError("GET /approvals: no approvals list", res.status);
    return body["approvals"] as Approval[];
  }

  /** `POST /approvals/{id}/decide`: approving runs the tool (with edited arguments, if any). */
  async decide<T = unknown>(id: string, decision: DecideInput, options: DecideOptions = {}): Promise<DecideResult<T>> {
    requireText("id", id);
    if (!isRecord(decision) || typeof decision.approved !== "boolean") {
      throw new TypeError("decision.approved must be a boolean");
    }
    const body: Record<string, unknown> = { approved: decision.approved };
    if (decision.editedArguments !== undefined) {
      if (!isRecord(decision.editedArguments)) throw new TypeError("decision.editedArguments must be a JSON object");
      body["edited_arguments"] = decision.editedArguments;
    }
    if (decision.reason !== undefined) body["reason"] = decision.reason;
    const what = "POST /approvals/{id}/decide";
    const res = await this.#http.request<unknown>({
      method: "POST",
      path: `/approvals/${encodeURIComponent(id)}/decide`,
      body,
      headers: { "idempotency-key": idempotencyKey(options.idempotencyKey) },
      signal: options.signal,
      idempotent: true,
    });
    const result = expectRecord(res, what);
    if (res.status === 200 && (result["status"] === "ok" || result["status"] === "denied")) {
      return result as DecideResult<T>;
    }
    throw new ToolProtocolError(
      `${what}: unexpected ${res.status} body status ${String(result["status"])}`,
      res.status,
    );
  }

  /**
   * `GET /runs/{run_id}/events`: the run's events until the server's `end` (not yielded).
   * Reconnects after a drop, or after `idleTimeoutMs` of silence, from the last event id, up to
   * `retry.maxRetries` times in a row without progress (a new event under a new id). Right after a
   * reconnect, events the server replays (same id and same content as one already yielded) are skipped.
   */
  async *events(runId: string, options: EventsOptions = {}): AsyncGenerator<RunEvent, void, undefined> {
    requireText("runId", runId);
    const { signal } = options;
    const path = `/runs/${encodeURIComponent(runId)}/events`;
    const closeOn = joinList(options.closeOn);
    let lastId = options.lastEventId ?? "";
    const yielded = new YieldedEvents();
    let connections = 0;
    let retryMs: number | undefined;
    for (let drops = 0; ; drops++) {
      const openedWith = lastId;
      // only a reconnect can replay what this generator already yielded; the first connection never skips
      let skipping = connections++ > 0;
      let fresh = 0;
      const stream = await this.#http.open({
        method: "GET",
        path,
        query: { close_on: closeOn },
        headers: lastId ? { "last-event-id": lastId } : {},
        accept: "text/event-stream",
        signal,
      });
      const parser = new SseParser({ maxSize: this.#maxEventSize, lastEventId: lastId });
      let dropCause: unknown;
      try {
        const { response } = stream;
        const media = (response.headers.get("content-type") ?? "").split(";")[0]?.trim().toLowerCase();
        if (media !== "text/event-stream" || response.body === null) {
          throw new ToolProtocolError(
            `GET ${path}: expected text/event-stream, got ${media || "nothing"}`,
            response.status,
          );
        }
        try {
          const reading = readEventStream(response.body, parser, { signal, idleTimeoutMs: this.#idleTimeoutMs });
          for await (const message of reading) {
            if (message.event === "end") return;
            if (skipping && message.ownId && yielded.has(message)) continue;
            skipping = false; // the first new event ends the replay
            if (message.ownId) yielded.add(message); // an inherited id must not replace its event's record
            fresh++;
            yield toRunEvent(message, path);
            signal?.throwIfAborted();
          }
        } catch (err) {
          if (signal?.aborted) throw signal.reason;
          if (err instanceof ToolProtocolError) throw err;
          dropCause = err; // the connection broke or went silent mid-stream: reconnect below
        }
      } finally {
        stream.close();
      }
      // WHATWG: the parser's id, including a reset by an empty `id` field and ids on data-less events
      lastId = parser.lastEventId;
      // progress needs a new event and a new id: a server that ignores Last-Event-ID and replays,
      // or flips between ids, would otherwise reconnect forever
      if (fresh > 0 && lastId !== openedWith) drops = 0;
      if (drops >= this.#http.policy.maxRetries) {
        throw new ToolTransportError(`GET ${path}: event stream closed before its end event`, "network", {
          cause: dropCause,
        });
      }
      retryMs = parser.retryMs ?? retryMs;
      const { baseDelayMs, maxDelayMs } = this.#http.policy;
      // a server's `retry: 0` must not turn reconnects into a busy loop
      const wait = Math.max(retryMs ?? this.#http.backoffMs(drops), baseDelayMs);
      await sleep(Math.min(wait, maxDelayMs), signal);
    }
  }

  /** `GET /runs/{run_id}/cost`: the run's ledger summary (404 when the server has no ledger). */
  async cost(runId: string, options: SignalOption = {}): Promise<LedgerSummary> {
    requireText("runId", runId);
    const res = await this.#http.request<LedgerSummary>({
      method: "GET",
      path: `/runs/${encodeURIComponent(runId)}/cost`,
      signal: options.signal,
    });
    return expectRecord(res, "GET /runs/{run_id}/cost");
  }

  /** Framework-neutral tool definitions whose `execute` invokes through this client. */
  toolDefinitions(options: ToolDefinitionsOptions = {}): Promise<ToolDefinition[]> {
    return toolDefinitions(this, options);
  }

  /** Vercel AI SDK tools: pass the SDK's `tool` and `jsonSchema` helpers. */
  asVercelTools(
    tool: (def: any) => any,
    jsonSchema: (schema: any) => any,
    options: ToolDefinitionsOptions = {},
  ): Promise<Record<string, unknown>> {
    return asVercelTools(this, tool, jsonSchema, options);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function expectRecord<T>(res: HttpResponse<T>, what: string): T & Record<string, unknown> {
  if (!isRecord(res.body)) throw new ToolProtocolError(`${what}: response is not a JSON object`, res.status);
  return res.body as T & Record<string, unknown>;
}

function requireText(name: string, value: unknown): void {
  if (typeof value !== "string" || value === "") throw new TypeError(`${name} must be a non-empty string`);
}

function idempotencyKey(given: string | undefined): string {
  if (given === undefined) return newIdempotencyKey();
  requireText("idempotencyKey", given);
  return given;
}

function joinList(values: string[] | undefined): string | undefined {
  return values && values.length > 0 ? values.join(",") : undefined;
}

function toRunEvent(message: SseMessage, path: string): RunEvent {
  let data: unknown;
  try {
    data = JSON.parse(message.data);
  } catch (err) {
    // contract §15.3: every data line is Event.to_dict() JSON; anything else is a broken stream
    throw new EventStreamError(`GET ${path}: event data is not JSON`, 200, { cause: err });
  }
  return message.id ? { id: message.id, event: message.event, data } : { event: message.event, data };
}
