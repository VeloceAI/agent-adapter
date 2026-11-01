/** One HTTP exchange with the tool server: auth, per-attempt timeout, abort, bounded retries, problems. */
import { readText, release, responseError } from "./body.ts";
import { ToolProtocolError, ToolServerError, ToolTransportError } from "./errors.ts";
import type { ClientOptions } from "./types.ts";

export const DEFAULT_TIMEOUT_MS: number = 30_000;
export const DEFAULT_MAX_RESPONSE_SIZE: number = 16 * 1024 * 1024;
export const MAX_TIMER_MS: number = 2_147_483_647;
// HTTP field values: HTAB, visible ASCII, SP and obs-text; anything else breaks the header
const TOKEN_CHARS = /^[\t\x20-\x7e\x80-\xff]*$/;
const ACCEPT_JSON = "application/json, application/problem+json";
// Safe methods: anything transient.
const GET_RETRY_STATUSES = new Set([408, 429, 502, 503, 504]);
// With Retry-After these mean "refused before running anything"; 502/504 may follow a tool run.
const POST_RETRY_STATUSES = new Set([429, 503]);
// The connection was never established, so the request cannot have reached the server.
const CONNECT_ERROR_CODES = new Set([
  "ECONNREFUSED",
  "ENOTFOUND",
  "EAI_AGAIN",
  "EHOSTUNREACH",
  "ENETUNREACH",
  "UND_ERR_CONNECT_TIMEOUT",
]);

export interface HttpRequest {
  method: "GET" | "POST";
  /** Already percent-encoded, starting with `/`. */
  path: string;
  query?: Record<string, string | undefined>;
  body?: unknown;
  headers?: Record<string, string>;
  accept?: string;
  signal?: AbortSignal | undefined;
  /** POST only: the request carries an `Idempotency-Key`, so refused attempts may be repeated. */
  idempotent?: boolean;
}

export interface HttpResponse<T> {
  status: number;
  headers: Headers;
  body: T;
}

/** An open streaming response; `close` aborts it and must always be called. */
export interface OpenStream {
  response: Response;
  close: () => void;
}

export interface RetryPolicy {
  maxRetries: number;
  baseDelayMs: number;
  maxDelayMs: number;
}

/** Shared transport of `ToolClient`; holds the token privately so it never shows in inspection. */
export class HttpClient {
  readonly policy: RetryPolicy;
  readonly #base: string;
  readonly #token: ClientOptions["token"];
  readonly #fetch: typeof fetch;
  readonly #timeoutMs: number;
  readonly #headers: Record<string, string>;
  readonly #maxResponseSize: number;

  constructor(options: ClientOptions) {
    this.#base = normaliseBaseUrl(options.baseUrl);
    this.#token = options.token;
    const custom = options.fetch;
    // always a plain call: invoked as a method, fetch gets this client as `this` and browsers throw
    // "Illegal invocation"; the global is resolved per call so a later polyfill or mock is used
    this.#fetch = custom ? (input, init) => custom(input, init) : (input, init) => globalThis.fetch(input, init);
    this.#timeoutMs = checkNumber("timeoutMs", options.timeoutMs ?? DEFAULT_TIMEOUT_MS, 1, MAX_TIMER_MS);
    this.#maxResponseSize = checkNumber(
      "maxResponseSize",
      options.maxResponseSize ?? DEFAULT_MAX_RESPONSE_SIZE,
      1,
      Number.MAX_SAFE_INTEGER,
    );
    this.#headers = { ...options.headers };
    const retry = options.retry === false ? { maxRetries: 0 } : (options.retry ?? {});
    const maxRetries = checkNumber("retry.maxRetries", retry.maxRetries ?? 2, 0, 100);
    if (!Number.isInteger(maxRetries)) throw new RangeError("retry.maxRetries must be an integer");
    this.policy = {
      maxRetries,
      baseDelayMs: checkNumber("retry.baseDelayMs", retry.baseDelayMs ?? 250, 0, MAX_TIMER_MS),
      maxDelayMs: checkNumber("retry.maxDelayMs", retry.maxDelayMs ?? 10_000, 0, MAX_TIMER_MS),
    };
  }

  /** Full-jitter exponential backoff for retry number `attempt` (0-based). */
  backoffMs(attempt: number): number {
    const ceiling = Math.min(this.policy.maxDelayMs, this.policy.baseDelayMs * 2 ** attempt);
    return Math.floor(Math.random() * ceiling);
  }

  /** Sends `request` and parses the JSON body of a 2xx response. */
  request<T>(request: HttpRequest): Promise<HttpResponse<T>> {
    return this.#run(request, false, async (response) => {
      const text = await readText(response, this.#maxResponseSize, label(request));
      let body: T;
      try {
        body = JSON.parse(text) as T;
      } catch (err) {
        throw new ToolProtocolError(`${label(request)}: response is not JSON`, response.status, { cause: err });
      }
      return { status: response.status, headers: response.headers, body };
    });
  }

  /** Sends `request` and returns the response once its headers arrive (the timeout stops there). */
  open(request: HttpRequest): Promise<OpenStream> {
    return this.#run(request, true, async (response, attempt) => {
      attempt.headersReceived();
      return { response, close: () => attempt.close() };
    });
  }

  async #run<T>(
    request: HttpRequest,
    keepOpen: boolean,
    finish: (response: Response, attempt: Attempt) => Promise<T>,
  ): Promise<T> {
    const url = this.#url(request);
    const body = request.body === undefined ? undefined : JSON.stringify(request.body);
    for (let attemptNo = 0; ; attemptNo++) {
      request.signal?.throwIfAborted();
      const headers = await this.#headersFor(request, body);
      // the caller may have aborted while the token provider ran: send nothing then
      request.signal?.throwIfAborted();
      // never follow a redirect: it would resend the bearer token and the body to another URL
      const init: RequestInit = { method: request.method, headers, redirect: "error" };
      if (body !== undefined) init.body = body;
      const attempt = new Attempt(this.#timeoutMs, request.signal);
      init.signal = attempt.signal;
      let pending: Promise<Response>;
      try {
        pending = this.#fetch(url, init);
      } catch (err) {
        // thrown before anything was sent (bad init, unusable fetch): a programming error, never a network one
        attempt.dispose();
        throw err;
      }
      let error: unknown;
      let retryAfterMs: number | undefined;
      let opened = false;
      try {
        const response = await pending;
        retryAfterMs = parseRetryAfter(response.headers.get("retry-after"));
        if (response.ok) {
          const value = await finish(response, attempt);
          opened = keepOpen;
          return value;
        }
        if (this.#mayRetry(request, attemptNo, response.status, retryAfterMs)) {
          await release(response);
          error = undefined;
        } else {
          error = await responseError(response, retryAfterMs, this.#maxResponseSize, label(request));
        }
      } catch (err) {
        error = attempt.translate(err, label(request));
      } finally {
        if (!opened) attempt.dispose();
      }
      if (error !== undefined && !this.#mayRetryError(request, attemptNo, error)) throw error;
      await sleep(retryAfterMs ?? this.backoffMs(attemptNo), request.signal);
    }
  }

  #mayRetry(request: HttpRequest, attemptNo: number, status: number, retryAfterMs: number | undefined): boolean {
    if (attemptNo >= this.policy.maxRetries) return false;
    if (retryAfterMs !== undefined && retryAfterMs > this.policy.maxDelayMs) return false;
    if (request.method === "GET") return GET_RETRY_STATUSES.has(status);
    return request.idempotent === true && POST_RETRY_STATUSES.has(status) && retryAfterMs !== undefined;
  }

  #mayRetryError(request: HttpRequest, attemptNo: number, error: unknown): boolean {
    if (attemptNo >= this.policy.maxRetries || !(error instanceof ToolTransportError)) return false;
    if (request.method === "GET") return true;
    return request.idempotent === true && error.reason === "network" && CONNECT_ERROR_CODES.has(error.code);
  }

  #url(request: HttpRequest): string {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(request.query ?? {})) {
      if (value !== undefined) params.set(key, value);
    }
    const query = params.toString();
    return `${this.#base}${request.path}${query ? `?${query}` : ""}`;
  }

  async #headersFor(request: HttpRequest, body: string | undefined): Promise<Headers> {
    const headers = new Headers(this.#headers);
    for (const [key, value] of Object.entries(request.headers ?? {})) headers.set(key, value);
    headers.set("accept", request.accept ?? ACCEPT_JSON);
    if (body !== undefined) headers.set("content-type", "application/json");
    const token = typeof this.#token === "function" ? await untilAborted(this.#token(), request.signal) : this.#token;
    if (token) {
      // the platform's own header error would quote the value, i.e. the token
      if (!TOKEN_CHARS.test(token)) throw new TypeError("invalid token characters");
      headers.set("authorization", `Bearer ${token}`);
    }
    return headers;
  }
}

/** One attempt's signal: aborted by the caller's signal, the timeout, or `close`. */
class Attempt {
  readonly signal: AbortSignal;
  readonly #controller = new AbortController();
  readonly #timeoutMs: number;
  readonly #user: AbortSignal | undefined;
  readonly #onAbort: () => void;
  #timer: ReturnType<typeof setTimeout> | undefined;
  #timedOut = false;

  constructor(timeoutMs: number, user: AbortSignal | undefined) {
    this.signal = this.#controller.signal;
    this.#timeoutMs = timeoutMs;
    this.#user = user;
    this.#onAbort = () => this.#controller.abort(user?.reason);
    if (user?.aborted) this.#controller.abort(user.reason);
    else user?.addEventListener("abort", this.#onAbort, { once: true });
    this.#timer = setTimeout(() => {
      this.#timedOut = true;
      this.#controller.abort(new DOMException("request timed out", "TimeoutError"));
    }, timeoutMs);
  }

  headersReceived(): void {
    clearTimeout(this.#timer);
    this.#timer = undefined;
  }

  /** Maps a failure of this attempt: the caller's abort reason, a timeout, or a network error. */
  translate(err: unknown, what: string): unknown {
    if (this.#user?.aborted) return this.#user.reason;
    if (this.#timedOut) {
      return new ToolTransportError(`${what} timed out after ${this.#timeoutMs} ms`, "timeout", { cause: err });
    }
    if (err instanceof ToolServerError || err instanceof ToolProtocolError) return err;
    // every rejection of fetch is a network error: runtimes word them differently (undici, browsers,
    // Deno, Workers, React Native) and only some carry a code; POSTs retry only on a connect-phase code
    const code = networkCode(err);
    const reason = err instanceof Error ? err.message : String(err);
    return new ToolTransportError(`${what} failed: ${reason}${code ? ` (${code})` : ""}`, "network", {
      cause: err,
      code,
    });
  }

  dispose(): void {
    clearTimeout(this.#timer);
    this.#user?.removeEventListener("abort", this.#onAbort);
  }

  close(): void {
    this.dispose();
    this.#controller.abort(new DOMException("stream closed", "AbortError"));
  }
}

function label(request: HttpRequest): string {
  return `${request.method} ${request.path}`;
}

function errorCode(value: unknown): string {
  if (typeof value === "object" && value !== null && "code" in value && typeof value.code === "string") {
    return value.code;
  }
  return "";
}

/** The platform code of a network failure: on the error, its cause, or the cause's first error (AggregateError). */
function networkCode(err: unknown): string {
  const cause = err instanceof Error ? err.cause : undefined;
  const first = cause instanceof AggregateError ? cause.errors[0] : undefined;
  return errorCode(cause) || errorCode(first) || errorCode(err);
}

/** `value` as a number in `[min, max]`, else a `RangeError` naming the option. */
export function checkNumber(name: string, value: number, min: number, max: number): number {
  if (typeof value !== "number" || Number.isNaN(value) || value < min || value > max) {
    throw new RangeError(`${name} must be a number from ${min} to ${max}`);
  }
  return value;
}

function normaliseBaseUrl(baseUrl: string): string {
  let url: URL;
  try {
    url = new URL(baseUrl);
  } catch {
    // neither the value nor the URL error (which carries it as `input`): it may hold credentials
    throw new TypeError("baseUrl is not a valid URL");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") throw new TypeError("baseUrl must be http(s)");
  if (url.username || url.password) throw new TypeError("baseUrl must not contain credentials; use `token`");
  if (url.search || url.hash) throw new TypeError("baseUrl must not have a query or fragment");
  return `${url.origin}${url.pathname.replace(/\/+$/, "")}`;
}

/** `Retry-After` in milliseconds: delay-seconds or an HTTP date; `undefined` when absent or invalid. */
export function parseRetryAfter(value: string | null, now: number = Date.now()): number | undefined {
  if (value === null) return undefined;
  const text = value.trim();
  if (/^[0-9]+$/.test(text)) return Number(text) * 1000;
  if (!/[a-z]/i.test(text)) return undefined;
  const at = Date.parse(text);
  return Number.isNaN(at) ? undefined : Math.max(0, at - now);
}

/** Waits `ms`; rejects with the signal's reason as soon as it aborts. */
export function sleep(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** Settles with `value`, or rejects with the signal's reason as soon as it aborts. */
function untilAborted<T>(value: T | Promise<T>, signal: AbortSignal | undefined): Promise<T> {
  if (signal === undefined) return Promise.resolve(value);
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(signal.reason);
      return;
    }
    const onAbort = (): void => reject(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    Promise.resolve(value).then(
      (result) => {
        signal.removeEventListener("abort", onAbort);
        resolve(result);
      },
      (err: unknown) => {
        signal.removeEventListener("abort", onAbort);
        reject(err);
      },
    );
  });
}

type RandomSource = Pick<Crypto, "getRandomValues"> & { randomUUID?: () => string };

/** A random v4 UUID for `Idempotency-Key`; falls back to `getRandomValues` outside secure contexts. */
export function newIdempotencyKey(source: RandomSource = globalThis.crypto): string {
  if (typeof source.randomUUID === "function") return source.randomUUID();
  const bytes = source.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6]! & 0x0f) | 0x40;
  bytes[8] = (bytes[8]! & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}
