/** Scripted `fetch` fake and response builders shared by the tests. */
import assert from "node:assert/strict";

import { HttpClient } from "../src/http.ts";
import type { ClientOptions, RunEvent } from "../src/index.ts";

export interface Recorded {
  url: URL;
  method: string;
  headers: Headers;
  body: string | undefined;
  signal: AbortSignal | undefined;
}

export type Step = Response | Error | ((request: Recorded) => Response | Promise<Response>);

export interface FakeFetch {
  fetch: typeof fetch;
  requests: Recorded[];
  /** Steps not consumed yet. */
  pending: () => number;
}

/** A fetch that answers each call with the next step (a Response, a thrown Error, or a function). */
export function fakeFetch(...steps: Step[]): FakeFetch {
  const queue = [...steps];
  const requests: Recorded[] = [];
  const impl = async (input: string | URL | Request, init: RequestInit = {}): Promise<Response> => {
    const request: Recorded = {
      url: new URL(String(input)),
      method: init.method ?? "GET",
      headers: new Headers(init.headers),
      body: typeof init.body === "string" ? init.body : undefined,
      signal: init.signal ?? undefined,
    };
    requests.push(request);
    if (request.signal?.aborted) throw request.signal.reason;
    const step = queue.shift();
    assert.ok(step !== undefined, `unexpected request ${request.method} ${request.url}`);
    if (step instanceof Error) throw step;
    return typeof step === "function" ? step(request) : step;
  };
  return { fetch: impl as typeof fetch, requests, pending: () => queue.length };
}

export function json(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

export function problem(
  status: number,
  kind: string,
  extra: Record<string, unknown> = {},
  headers: Record<string, string> = {},
): Response {
  const body = { type: `urn:agent-adapter:error:${kind}`, title: kind, status, detail: `detail of ${kind}`, ...extra };
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/problem+json", ...headers },
  });
}

/** A network failure as Node's fetch reports it (`TypeError` with a coded cause). */
export function networkError(code: string): TypeError {
  const cause = Object.assign(new Error(code), { code });
  return new TypeError("fetch failed", { cause });
}

/** Never answers; rejects with the signal's reason once the request is aborted. */
export function hang(request: Recorded): Promise<Response> {
  return new Promise((_resolve, reject) => {
    const signal = request.signal;
    assert.ok(signal, "request has no signal");
    signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  });
}

/** An SSE response whose body yields `chunks` (strings are UTF-8 encoded) then closes, or errors with `fail`. */
export function sse(chunks: Array<string | Uint8Array>, options: { fail?: Error; status?: number } = {}): Response {
  const encoder = new TextEncoder();
  let index = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      const chunk = chunks[index++];
      if (chunk === undefined) {
        if (options.fail) controller.error(options.fail);
        else controller.close();
        return;
      }
      controller.enqueue(typeof chunk === "string" ? encoder.encode(chunk) : chunk);
    },
  });
  return new Response(body, { status: options.status ?? 200, headers: { "content-type": "text/event-stream" } });
}

/** An SSE response that sends `chunks` and then stays open until cancelled; `cancelled` resolves on cancel. */
export function openSse(chunks: string[]): { response: Response; cancelled: Promise<void> } {
  const encoder = new TextEncoder();
  let onCancel: () => void = () => {};
  const cancelled = new Promise<void>((resolve) => {
    onCancel = resolve;
  });
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
    },
    cancel() {
      onCancel();
    },
  });
  return {
    response: new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }),
    cancelled,
  };
}

/** Fast retries for tests. */
export const FAST_RETRY = { maxRetries: 2, baseDelayMs: 1, maxDelayMs: 50 } as const;

/** One SSE frame carrying an `Event.to_dict()` payload, as the server sends it. */
export function frame(id: number, type: string, data: Record<string, unknown> = {}): string {
  const event = { type, run_id: "run_1", call_id: "call_1", parent_id: "", ts: 1, data };
  return `id: ${id}\nevent: ${type}\ndata: ${JSON.stringify(event)}\n\n`;
}

export const END = "event: end\ndata: {}\n\n";

export async function collect(stream: AsyncIterable<RunEvent>): Promise<RunEvent[]> {
  const out: RunEvent[] = [];
  for await (const event of stream) out.push(event);
  return out;
}

/** An `HttpClient` over a scripted fake fetch with fast retries. */
export function http(steps: Step[], options: Partial<ClientOptions> = {}): { client: HttpClient; fake: FakeFetch } {
  const fake = fakeFetch(...steps);
  const client = new HttpClient({ baseUrl: "http://srv/api/", fetch: fake.fetch, retry: FAST_RETRY, ...options });
  return { client, fake };
}

/** The error `promise` rejects with; fails the test if it resolves. */
export async function rejection(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (err) {
    assert.ok(err instanceof Error);
    return err;
  }
  assert.fail("expected a rejection");
}
