import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { ToolClient, ToolTransportError } from "../src/index.ts";
import type { RunEvent } from "../src/index.ts";
import { collect, END, frame, sse } from "./helpers.ts";

/** A fetch answering connection n with `bodies(n)` (1-based); after `cap` connections it sends `end`. */
function server(bodies: (n: number) => string[], cap = 50) {
  let calls = 0;
  const fetch = async (): Promise<Response> => (++calls > cap ? sse([END]) : sse(bodies(calls)));
  return { fetch: fetch as typeof globalThis.fetch, calls: () => calls };
}

function client(fetch: typeof globalThis.fetch): ToolClient {
  return new ToolClient({ baseUrl: "http://srv", fetch, retry: { maxRetries: 2, baseDelayMs: 1, maxDelayMs: 20 } });
}

const ids = (events: RunEvent[]): Array<string | undefined> => events.map((e) => e.id);
const custom = (id: string, n: number): string =>
  `id: ${id}\nevent: custom.step\ndata: {"type":"custom.step","n":${n}}\n\n`;

describe("events dedupe never drops new events", () => {
  test("a server that renumbers after a restart loses nothing", async () => {
    const s = server((n) =>
      n === 1
        ? [frame(57, "tool.started"), frame(58, "tool.retry")]
        : [frame(1, "tool.started"), frame(2, "tool.retry"), frame(3, "tool.finished"), END],
    );
    assert.deepEqual(ids(await collect(client(s.fetch).events("run_1"))), ["57", "58", "1", "2", "3"]);
  });

  test("decreasing ids on the first connection are all yielded", async () => {
    const s = server(() => [frame(3, "tool.started"), frame(2, "tool.retry"), frame(1, "tool.finished"), END]);
    assert.deepEqual(ids(await collect(client(s.fetch).events("run_1"))), ["3", "2", "1"]);
  });

  test("one constant non-numeric id is not a replay", async () => {
    const s = server(() => [custom("x", 1), custom("x", 2), custom("x", 3), END]);
    const events = await collect(client(s.fetch).events("run_1"));
    assert.deepEqual(
      events.map((e) => (e.data as { n: number }).n),
      [1, 2, 3],
    );
  });

  test("the same id with new content after a reconnect is yielded", async () => {
    const s = server((n) => (n === 1 ? [custom("x", 1)] : [custom("x", 2), END]));
    const events = await collect(client(s.fetch).events("run_1"));
    assert.deepEqual(
      events.map((e) => (e.data as { n: number }).n),
      [1, 2],
    );
  });

  test("skipping stops at the first event that is not a replay", async () => {
    const s = server((n) =>
      n === 1
        ? [frame(1, "tool.started"), frame(2, "tool.retry")]
        : [frame(1, "tool.started"), frame(5, "tool.finished"), frame(2, "tool.retry"), END],
    );
    assert.deepEqual(ids(await collect(client(s.fetch).events("run_1"))), ["1", "2", "5", "2"]);
  });

  test("ids alternating between two values end in ToolTransportError after maxRetries", async () => {
    const s = server((n) => [n % 2 === 1 ? frame(3, "tool.started") : frame(2, "tool.retry")]);
    const seen: RunEvent[] = [];
    await assert.rejects(async () => {
      for await (const event of client(s.fetch).events("run_1")) seen.push(event);
    }, ToolTransportError);
    assert.deepEqual(ids(seen), ["3", "2"]);
    assert.equal(s.calls(), 4);
  });
});

describe("events dedupe after inherited ids", () => {
  test("an event that only inherits an id does not replace the replay record of that id", async () => {
    const inherited = `event: tool.retry
data: {"type":"tool.retry"}

`;
    const s = server((n) =>
      n === 1 ? [frame(5, "tool.started"), inherited] : [frame(5, "tool.started"), frame(6, "tool.finished"), END],
    );
    const events = await collect(client(s.fetch).events("run_1"));
    assert.deepEqual(
      events.map((e) => [e.id, e.event]),
      [
        ["5", "tool.started"],
        ["5", "tool.retry"],
        ["6", "tool.finished"],
      ],
    );
  });
});

describe("events dedupe memory", () => {
  test("an id longer than 256 characters is not remembered, so its replay is yielded again", async () => {
    const long = "x".repeat(257);
    const s = server((n) => (n === 1 ? [custom(long, 1)] : [custom(long, 1), custom("y", 2), END]));
    const events = await collect(client(s.fetch).events("run_1"));
    assert.deepEqual(
      events.map((e) => (e.data as { n: number }).n),
      [1, 1, 2],
    );
  });

  test("a 256-character id is still remembered and its replay skipped", async () => {
    const edge = "x".repeat(256);
    const s = server((n) => (n === 1 ? [custom(edge, 1)] : [custom(edge, 1), custom("y", 2), END]));
    const events = await collect(client(s.fetch).events("run_1"));
    assert.deepEqual(
      events.map((e) => (e.data as { n: number }).n),
      [1, 2],
    );
  });

  test("only the last 1024 ids are remembered; an older replay is yielded again", async () => {
    const many = Array.from({ length: 1030 }, (_, i) => frame(i + 1, "tool.started"));
    const s = server((n) => (n === 1 ? many : [frame(1, "tool.started"), frame(1030, "tool.started"), END]));
    const events = await collect(client(s.fetch).events("run_1"));
    assert.equal(events.length, 1032);
    assert.deepEqual(ids(events.slice(-2)), ["1", "1030"]);
  });
});
