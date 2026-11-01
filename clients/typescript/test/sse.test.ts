import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { EventStreamError, ToolTransportError } from "../src/index.ts";
import { readEventStream, SseParser } from "../src/sse.ts";
import type { SseMessage } from "../src/sse.ts";
import { sse } from "./helpers.ts";

type Plain = Omit<SseMessage, "ownId">;

/** The message without `ownId`, which only the "own id" test looks at. */
const plain = ({ id, event, data }: SseMessage): Plain => ({ id, event, data });

function parseAll(...chunks: string[]): Plain[] {
  const parser = new SseParser();
  return chunks.flatMap((chunk) => parser.feed(chunk)).map(plain);
}

const STREAM =
  'id: 1\nevent: tool.started\ndata: {"tool":"add"}\n\n' +
  ": keep-alive\n\n" +
  "id: 2\r\nevent: tool.finished\r\ndata: line one\r\ndata: line two\r\n\r\n" +
  "id: 3\revent: end\rdata: {}\r\r";

const EXPECTED: Plain[] = [
  { id: "1", event: "tool.started", data: '{"tool":"add"}' },
  { id: "2", event: "tool.finished", data: "line one\nline two" },
  { id: "3", event: "end", data: "{}" },
];

describe("SseParser", () => {
  test("parses frames with LF, CRLF and CR line ends and skips comments", () => {
    assert.deepEqual(parseAll(STREAM), EXPECTED);
  });

  test("gives the same result for a split at every position", () => {
    for (let i = 0; i <= STREAM.length; i++) {
      assert.deepEqual(parseAll(STREAM.slice(0, i), STREAM.slice(i)), EXPECTED, `split at ${i}`);
    }
  });

  test("gives the same result one character at a time", () => {
    assert.deepEqual(parseAll(...STREAM.split("")), EXPECTED);
  });

  test("CR at the end of a chunk followed by LF is one line end", () => {
    assert.deepEqual(parseAll("data: a\r", "\ndata: b\r\n\r\n"), [{ id: "", event: "message", data: "a\nb" }]);
  });

  test("defaults the event type to message and strips one leading space only", () => {
    assert.deepEqual(parseAll("data:  two spaces\ndata:none\n\n"), [
      { id: "", event: "message", data: " two spaces\nnone" },
    ]);
  });

  test("a field without a colon has an empty value", () => {
    assert.deepEqual(parseAll("data\ndata\n\n"), [{ id: "", event: "message", data: "\n" }]);
  });

  test("an event without data is not dispatched and its type is reset", () => {
    assert.deepEqual(parseAll("event: lonely\n\ndata: x\n\n"), [{ id: "", event: "message", data: "x" }]);
  });

  test("ids persist, an empty id resets, an id with NUL is ignored", () => {
    assert.deepEqual(parseAll("id: 7\ndata: a\n\ndata: b\n\nid: x\u0000y\ndata: c\n\nid\ndata: d\n\n"), [
      { id: "7", event: "message", data: "a" },
      { id: "7", event: "message", data: "b" },
      { id: "7", event: "message", data: "c" },
      { id: "", event: "message", data: "d" },
    ]);
  });

  test("retry takes ASCII digits only; unknown fields are ignored", () => {
    const parser = new SseParser();
    parser.feed("retry: 1500\nfoo: bar\n\n");
    assert.equal(parser.retryMs, 1500);
    parser.feed("retry: 2s\n\n");
    assert.equal(parser.retryMs, 1500);
    parser.feed("id: 9\n");
    assert.equal(parser.lastEventId, "", "set when the event is dispatched");
    parser.feed("\n");
    assert.equal(parser.lastEventId, "9", "a blank line dispatches the id even without data");
  });

  test("marks events that carry their own id field", () => {
    const parser = new SseParser({ lastEventId: "4" });
    const stream = "data: a\n\nid: 5\ndata: b\n\ndata: c\n\nid\ndata: d\n\n";
    assert.deepEqual(
      parser.feed(stream).map((m) => [m.id, m.ownId]),
      [
        ["4", false],
        ["5", true],
        ["5", false],
        ["", true],
      ],
    );
  });

  test("starts from a given last event id", () => {
    const parser = new SseParser({ lastEventId: "4" });
    assert.equal(parser.lastEventId, "4");
    assert.deepEqual(parser.feed("data: a\n\n").map(plain), [{ id: "4", event: "message", data: "a" }]);
  });

  test("strips one leading BOM", () => {
    const bom = String.fromCharCode(0xfeff);
    assert.deepEqual(parseAll(bom, "data: a\n\n"), [{ id: "", event: "message", data: "a" }]);
    assert.deepEqual(parseAll(`${bom}${bom}data: a\n\n`), []);
  });

  test("an unterminated event at the end is discarded", () => {
    assert.deepEqual(parseAll("data: a\n\ndata: half"), [{ id: "", event: "message", data: "a" }]);
    assert.deepEqual(parseAll("data: a\r"), []);
  });

  test("rejects a line longer than maxSize, even when it arrives in pieces", () => {
    const parser = new SseParser({ maxSize: 10 });
    parser.feed("data: 1234");
    assert.throws(() => parser.feed("56789"), EventStreamError);
  });

  test("rejects an event whose data grows past maxSize", () => {
    const parser = new SseParser({ maxSize: 10 });
    assert.throws(() => parser.feed("data:1234\ndata:1234\ndata:1234\n"), EventStreamError);
  });

  test("accepts a line exactly at maxSize", () => {
    const parser = new SseParser({ maxSize: 10 });
    assert.deepEqual(parser.feed("data:12345\n\n").map(plain), [{ id: "", event: "message", data: "12345" }]);
  });

  test("rejects a non-positive maxSize", () => {
    assert.throws(() => new SseParser({ maxSize: 0 }), RangeError);
  });

  test("a huge line arriving in small chunks parses in linear time", () => {
    const parser = new SseParser({ maxSize: 2_000_000 });
    const payload = "x".repeat(1_000_000);
    const started = performance.now();
    const line = `data: ${payload}`;
    for (let i = 0; i < line.length; i += 1000) parser.feed(line.slice(i, i + 1000));
    const [message] = parser.feed("\n\n");
    assert.equal(message?.data.length, payload.length);
    assert.ok(performance.now() - started < 2000, "parsing was not linear");
  });
});

describe("readEventStream", () => {
  test("decodes UTF-8 split inside a character", async () => {
    const bytes = new TextEncoder().encode("data: café ☕\n\n");
    const chunks = Array.from(bytes, (b) => new Uint8Array([b]));
    const response = sse(chunks);
    const out: Plain[] = [];
    for await (const message of readEventStream(response.body!, new SseParser())) out.push(plain(message));
    assert.deepEqual(out, [{ id: "", event: "message", data: "café ☕" }]);
  });

  test("propagates a body error", async () => {
    const response = sse(["data: a\n\n"], { fail: new Error("boom") });
    const out: string[] = [];
    await assert.rejects(async () => {
      for await (const message of readEventStream(response.body!, new SseParser())) out.push(message.data);
    }, /boom/);
    assert.deepEqual(out, ["a"]);
  });

  test("an abort cancels a pending read and rejects with the reason", async () => {
    const controller = new AbortController();
    const body = new ReadableStream<Uint8Array>({
      cancel() {
        throw new Error("source refuses to cancel");
      },
    });
    setTimeout(() => controller.abort(new Error("stop reading")), 5);
    await assert.rejects(async () => {
      for await (const _message of readEventStream(body, new SseParser(), { signal: controller.signal })) {
        assert.fail("no events expected");
      }
    }, /stop reading/);
  });

  test("an already aborted signal rejects before reading", async () => {
    const body = new ReadableStream<Uint8Array>();
    await assert.rejects(async () => {
      for await (const _message of readEventStream(body, new SseParser(), {
        signal: AbortSignal.abort(new Error("early")),
      })) {
        assert.fail("no events expected");
      }
    }, /early/);
    assert.equal(body.locked, false);
  });

  test("an idle stream is cancelled and raises a timeout", async () => {
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(": keep-alive\n\n"));
      },
      cancel() {
        cancelled = true;
      },
    });
    await assert.rejects(
      async () => {
        for await (const _message of readEventStream(body, new SseParser(), { idleTimeoutMs: 20 })) {
          assert.fail("no events expected");
        }
      },
      (err: unknown) => err instanceof ToolTransportError && err.reason === "timeout",
    );
    assert.ok(cancelled);
  });

  test("a slow consumer does not count as idle time", async () => {
    const body = sse(['data: "a"\n\n', 'data: "b"\n\n']);
    const seen: string[] = [];
    for await (const message of readEventStream(body.body!, new SseParser(), { idleTimeoutMs: 20 })) {
      seen.push(message.data);
      await new Promise((resolve) => setTimeout(resolve, 40));
    }
    assert.deepEqual(seen, ['"a"', '"b"']);
  });

  test("cancels the body when the consumer stops early", async () => {
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("data: a\n\ndata: b\n\n"));
      },
      cancel() {
        cancelled = true;
      },
    });
    for await (const message of readEventStream(body, new SseParser())) {
      assert.equal(message.data, "a");
      break;
    }
    assert.ok(cancelled);
  });
});
