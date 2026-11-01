import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { ToolClient, ToolTransportError } from "../src/index.ts";
import type { RetryOptions, RunEvent } from "../src/index.ts";
import { collect, END, frame, sse } from "./helpers.ts";

/** A server (or proxy) that ignores Last-Event-ID: every connection replays `body` and closes; ends after `cap`. */
function replaying(body: string[], cap = 50) {
  let calls = 0;
  const fetch = async (): Promise<Response> => (++calls > cap ? sse([END]) : sse(body));
  return { fetch: fetch as typeof globalThis.fetch, calls: () => calls };
}

function client(
  fetch: typeof globalThis.fetch,
  retry: RetryOptions = { maxRetries: 2, baseDelayMs: 1, maxDelayMs: 50 },
) {
  return new ToolClient({ baseUrl: "http://srv", fetch, retry });
}

describe("events against a replaying server", () => {
  test("a replay with retry: 0 gives up after maxRetries reconnects and yields each event once", async () => {
    const server = replaying(["retry: 0\n\n", frame(1, "tool.started")]);
    const seen: RunEvent[] = [];
    await assert.rejects(
      async () => {
        for await (const event of client(server.fetch).events("run_1")) seen.push(event);
      },
      (err: unknown) => err instanceof ToolTransportError && err.reason === "network",
    );
    assert.equal(server.calls(), 3);
    assert.deepEqual(
      seen.map((e) => e.id),
      ["1"],
    );
  });

  test("a replay without ids is bounded too", async () => {
    const server = replaying(['data: {"type":"x"}\n\n']);
    await assert.rejects(collect(client(server.fetch).events("run_1")), ToolTransportError);
    assert.equal(server.calls(), 3);
  });

  test("the reconnect wait never drops below baseDelayMs, even with retry: 0", async () => {
    const server = replaying(["retry: 0\n\n", frame(1, "tool.started")]);
    const started = performance.now();
    await assert.rejects(
      collect(client(server.fetch, { maxRetries: 2, baseDelayMs: 30, maxDelayMs: 100 }).events("run_1")),
      ToolTransportError,
    );
    assert.ok(performance.now() - started >= 55, "two reconnects waited at least baseDelayMs each");
  });

  test("events replayed after a reconnect are not yielded twice", async () => {
    const responses = [
      sse([frame(1, "tool.started"), frame(2, "tool.retry")]),
      sse([frame(1, "tool.started"), frame(2, "tool.retry"), frame(3, "tool.finished"), END]),
    ];
    const fetch = (async () => responses.shift()!) as typeof globalThis.fetch;
    const events = await collect(client(fetch).events("run_1"));
    assert.deepEqual(
      events.map((e) => e.id),
      ["1", "2", "3"],
    );
  });

  test("new events keep resetting the reconnect budget", async () => {
    let id = 0;
    let calls = 0;
    const fetch = (async () =>
      ++calls > 5 ? sse([END]) : sse([frame(++id, "tool.started")])) as typeof globalThis.fetch;
    const events = await collect(client(fetch).events("run_1"));
    assert.equal(events.length, 5);
    assert.equal(calls, 6);
  });
});
