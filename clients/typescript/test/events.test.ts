import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { EventStreamError, ForbiddenError, ToolClient, ToolProtocolError, ToolTransportError } from "../src/index.ts";
import type { ClientOptions } from "../src/index.ts";
import { collect, END, FAST_RETRY, fakeFetch, frame, json, networkError, openSse, problem, sse } from "./helpers.ts";
import type { Step } from "./helpers.ts";

function client(steps: Step[], options: Partial<ClientOptions> = {}) {
  const fake = fakeFetch(...steps);
  const tc = new ToolClient({ baseUrl: "http://srv", fetch: fake.fetch, retry: FAST_RETRY, ...options });
  return { tc, fake };
}

describe("events", () => {
  test("requests the run's stream with close_on and Last-Event-ID", async () => {
    const { tc, fake } = client([sse([END])]);
    await collect(tc.events("run 1", { closeOn: ["tool.finished", "tool.failed"], lastEventId: "4" }));
    const req = fake.requests[0]!;
    assert.equal(req.url.pathname, "/runs/run%201/events");
    assert.equal(req.url.searchParams.get("close_on"), "tool.finished,tool.failed");
    assert.equal(req.headers.get("last-event-id"), "4");
    assert.equal(req.headers.get("accept"), "text/event-stream");
  });

  test("yields parsed events, skips keep-alives and stops at end", async () => {
    const body = `${frame(1, "tool.started", { tool: "add" })}: keep-alive\n\n${frame(2, "tool.finished")}${END}`;
    const { tc } = client([sse([body.slice(0, 7), body.slice(7, 90), body.slice(90)])]);
    const events = await collect(tc.events("run_1"));
    assert.deepEqual(
      events.map((e) => [e.id, e.event]),
      [
        ["1", "tool.started"],
        ["2", "tool.finished"],
      ],
    );
    assert.deepEqual((events[0]!.data as { data: unknown }).data, { tool: "add" });
  });

  test("omits an empty id", async () => {
    const { tc } = client([sse(['event: custom.note\ndata: {"type":"custom.note"}\n\n', END])]);
    const [event] = await collect(tc.events("run_1"));
    assert.deepEqual(event, { event: "custom.note", data: { type: "custom.note" } });
  });

  test("non-JSON event data is a stream error, not retried", async () => {
    const { tc, fake } = client([sse(["event: custom.note\ndata: plain text\n\n", END]), sse([END])]);
    await assert.rejects(collect(tc.events("run_1")), EventStreamError);
    assert.equal(fake.requests.length, 1);
  });

  test("an empty id field resets the id sent on reconnect", async () => {
    const { tc, fake } = client([sse([frame(5, "tool.started"), 'id\ndata: {"type":"x"}\n\n']), sse([END])]);
    await collect(tc.events("run_1"));
    assert.equal(fake.requests[1]!.headers.has("last-event-id"), false);
  });

  test("an id on an event without data still counts for the reconnect", async () => {
    const { tc, fake } = client([sse(["id: 7\n\n"]), sse([END])]);
    await collect(tc.events("run_1"));
    assert.equal(fake.requests[1]!.headers.get("last-event-id"), "7");
  });

  test("a given lastEventId is kept when the server sends no ids", async () => {
    const { tc, fake } = client([sse(['data: {"type":"x"}\n\n']), sse([END])]);
    const events = await collect(tc.events("run_1", { lastEventId: "4" }));
    assert.equal(events[0]!.id, "4");
    assert.equal(fake.requests[1]!.headers.get("last-event-id"), "4");
  });

  test("a silent half-open stream times out and reconnects from the last id", { timeout: 5000 }, async () => {
    const open = openSse([frame(1, "tool.started"), ": keep-alive\n\n"]);
    const { tc, fake } = client([open.response, sse([END])], { idleTimeoutMs: 30 });
    const events = await collect(tc.events("run_1"));
    assert.deepEqual(
      events.map((e) => e.id),
      ["1"],
    );
    await open.cancelled;
    assert.equal(fake.requests[0]!.signal?.aborted, true);
    assert.equal(fake.requests[1]!.headers.get("last-event-id"), "1");
  });

  test("keep-alives reset the idle timer", async () => {
    const encoder = new TextEncoder();
    let sent = 0;
    const body = new ReadableStream<Uint8Array>({
      async pull(controller) {
        await new Promise((resolve) => setTimeout(resolve, 10));
        if (sent++ < 6) controller.enqueue(encoder.encode(": keep-alive\n\n"));
        else {
          controller.enqueue(encoder.encode(END));
          controller.close();
        }
      },
    });
    const response = new Response(body, { headers: { "content-type": "text/event-stream" } });
    const { tc, fake } = client([response], { idleTimeoutMs: 40 });
    assert.deepEqual(await collect(tc.events("run_1")), []);
    assert.equal(fake.requests.length, 1);
  });

  test("rejects a bad idleTimeoutMs", () => {
    assert.throws(() => new ToolClient({ baseUrl: "http://srv", idleTimeoutMs: 0 }), RangeError);
  });

  test("rejects a response that is not an event stream", async () => {
    const { tc } = client([json(200, {})]);
    await assert.rejects(collect(tc.events("run_1")), ToolProtocolError);
  });

  test("raises the typed error of a refused stream", async () => {
    const { tc } = client([problem(403, "forbidden")]);
    await assert.rejects(collect(tc.events("run_1")), ForbiddenError);
  });

  test("reconnects after a drop from the last event id", async () => {
    const { tc, fake } = client([sse([frame(1, "tool.started")]), sse([frame(2, "tool.finished"), END])]);
    const events = await collect(tc.events("run_1"));
    assert.deepEqual(
      events.map((e) => e.id),
      ["1", "2"],
    );
    assert.equal(fake.requests[0]!.headers.has("last-event-id"), false);
    assert.equal(fake.requests[1]!.headers.get("last-event-id"), "1");
  });

  test("reconnects after the body fails mid-stream", async () => {
    const { tc, fake } = client([
      sse([frame(1, "tool.started")], { fail: new TypeError("terminated") }),
      sse([frame(2, "tool.finished"), END]),
    ]);
    assert.equal((await collect(tc.events("run_1"))).length, 2);
    assert.equal(fake.requests[1]!.headers.get("last-event-id"), "1");
  });

  test("honours the server's retry field within maxDelayMs", async () => {
    const { tc } = client([sse(["retry: 1\n\n", frame(1, "tool.started")]), sse([END])]);
    assert.equal((await collect(tc.events("run_1"))).length, 1);
  });

  test("gives up after maxRetries drops without progress", async () => {
    const { tc, fake } = client([sse([]), sse([]), sse([])]);
    await assert.rejects(collect(tc.events("run_1")), (err: unknown) => {
      assert.ok(err instanceof ToolTransportError);
      assert.equal(err.reason, "network");
      return true;
    });
    assert.equal(fake.requests.length, 3);
  });

  test("with retry: false a drop is an error at once", async () => {
    const { tc, fake } = client([sse([frame(1, "tool.started")])], { retry: false });
    const seen: string[] = [];
    await assert.rejects(async () => {
      for await (const event of tc.events("run_1")) seen.push(event.event);
    }, ToolTransportError);
    assert.deepEqual(seen, ["tool.started"]);
    assert.equal(fake.requests.length, 1);
  });

  test("a failed reconnect raises its error", async () => {
    const { tc } = client([sse([frame(1, "tool.started")]), problem(403, "forbidden")]);
    await assert.rejects(collect(tc.events("run_1")), ForbiddenError);
  });

  test("an oversized event fails without reconnecting", async () => {
    const { tc, fake } = client([sse([`data: ${"x".repeat(200)}\n\n`])], { maxEventSize: 100 });
    await assert.rejects(collect(tc.events("run_1")), EventStreamError);
    assert.equal(fake.requests.length, 1);
  });

  test("breaking out of the loop cancels the stream", async () => {
    const open = openSse([frame(1, "tool.started"), frame(2, "tool.finished")]);
    const { tc, fake } = client([open.response]);
    for await (const event of tc.events("run_1")) {
      assert.equal(event.id, "1");
      break;
    }
    await open.cancelled;
    assert.equal(fake.requests[0]!.signal?.aborted, true);
  });

  test("aborting the signal ends the loop with its reason", async () => {
    const controller = new AbortController();
    const open = openSse([frame(1, "tool.started")]);
    const { tc } = client([open.response]);
    await assert.rejects(async () => {
      for await (const _event of tc.events("run_1", { signal: controller.signal })) {
        controller.abort(new Error("enough"));
      }
    }, /enough/);
  });

  test("a network failure before the stream opens is retried like any GET", async () => {
    const { tc, fake } = client([networkError("ECONNRESET"), sse([END])]);
    assert.deepEqual(await collect(tc.events("run_1")), []);
    assert.equal(fake.requests.length, 2);
  });

  test("rejects an empty run id", async () => {
    const { tc } = client([]);
    await assert.rejects(collect(tc.events("")), TypeError);
  });
});
