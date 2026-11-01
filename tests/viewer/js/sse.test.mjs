// chatapi.js: the SSE parser and the turn stream (chunk splits, heartbeats, reconnect with since).
import assert from "node:assert/strict";
import { test } from "node:test";

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { loadDom } from "./fakedom.mjs";
import { STATIC } from "./load.mjs";

const warnings = [];
console.warn = (...args) => warnings.push(args.join(" ")); // reconnect notices, asserted below
const { AA } = loadDom({ hash: "#token=tok%2Fen&run=x" }, "data.js", "chatapi.js");
AA.csrf = "csrf-1";

const frame = (seq, type, data) => "id: " + seq + "\nevent: " + type + "\ndata: " + JSON.stringify({ seq, type, ...data }) + "\n\n";

function parseAll(chunks) {
  const events = [];
  const comments = [];
  const parser = AA.sse.createParser((e) => events.push(e), (c) => comments.push(c));
  chunks.forEach((c) => parser.feed(c));
  return { events, comments };
}

test("frames split at every possible point parse the same", () => {
  const text = ": ping\n\n" + frame(0, "text", { text: "héllo 🌍" }) + ": ping\n\n" + frame(1, "done", { status: "done" });
  const whole = parseAll([text]);
  assert.equal(whole.events.length, 2);
  assert.deepEqual(whole.comments, ["ping", "ping"]);
  for (let cut = 1; cut < text.length; cut += 1) {
    const split = parseAll([text.slice(0, cut), text.slice(cut)]);
    assert.deepEqual(split.events, whole.events, "cut at " + cut);
  }
  assert.deepEqual(whole.events[0], { id: "0", event: "text", data: JSON.stringify({ seq: 0, type: "text", text: "héllo 🌍" }) });
});

test("CRLF and CR line ends, multi-line data, and frames without data", () => {
  const { events } = parseAll(["id: 7\r\nevent: a\r\ndata: one\r\ndata: two\r", "\n\r\nevent: empty\n\ndata:x\r\rretry: 5\n\n"]);
  assert.deepEqual(events, [
    { id: "7", event: "a", data: "one\ntwo" },
    { id: null, event: "message", data: "x" },
  ]);
});

// A fetch whose responses stream the given byte chunks; `script` = [{status, chunks, error?}].
function fakeFetch(script) {
  const calls = [];
  const fetch = async (url, options) => {
    calls.push({ url, headers: options.headers });
    const step = script.shift();
    if (!step) throw new Error("no more responses");
    if (step.throws) throw new Error(step.throws);
    const queue = step.chunks.map((c) => (typeof c === "string" ? new TextEncoder().encode(c) : c));
    return {
      ok: step.status === 200, status: step.status,
      json: async () => step.body || {},
      body: { getReader: () => ({
        read: async () => {
          if (queue.length) return { done: false, value: queue.shift() };
          if (step.error) throw new Error(step.error);
          return { done: true, value: undefined };
        },
        cancel: async () => {},
      }) },
    };
  };
  return { fetch, calls };
}

// Timers run at once (in order), so reconnects need no waiting.
const instant = { setTimeout: (fn) => { Promise.resolve().then(fn); return 1; }, clearTimeout() {} };
const noTimers = { setTimeout: () => 0, clearTimeout() {} };

function run(script, deps) {
  return new Promise((resolve) => {
    const seen = [];
    const statuses = [];
    const fake = fakeFetch(script);
    const result = { seen, statuses, calls: fake.calls };
    AA.chatApi.streamTurn("turn_1", 0, {
      onEvent: (ev) => { seen.push(ev); if (ev.type === "done" || ev.type === "error") resolve(result); },
      onStatus: (s) => statuses.push(s),
      onFatal: (err) => { result.fatal = err; resolve(result); },
    }, Object.assign({ fetch: fake.fetch }, instant, deps || {}));
  });
}

test("a stream delivers events in order with the token header, split mid-character", async () => {
  const bytes = new TextEncoder().encode(frame(0, "text", { text: "ü€" }) + frame(1, "done", { status: "done" }));
  const chunks = Array.from(bytes, (b) => Uint8Array.of(b)); // one byte per read
  const out = await run([{ status: 200, chunks }]);
  assert.deepEqual(out.seen.map((e) => e.type), ["text", "done"]);
  assert.equal(out.seen[0].text, "ü€");
  assert.equal(out.calls.length, 1);
  assert.equal(out.calls[0].url, "api/chat/turns/turn_1/stream?since=0");
  assert.equal(out.calls[0].headers["X-AA-Token"], "tok/en", "the token read from the address");
});

test("a dropped stream reconnects from the next seq and skips repeats", async () => {
  const out = await run([
    { status: 200, chunks: [": ping\n\n", frame(0, "text", { text: "a" }), frame(1, "tool", { call_id: "c", status: "running" })], error: "reset" },
    { status: 503, chunks: [] },
    { status: 200, chunks: [frame(1, "tool", { call_id: "c", status: "running" }), frame(2, "text", { text: "b" }), "not a frame\n\n", frame(3, "done", { status: "done" })] },
  ]);
  assert.deepEqual(out.seen.map((e) => e.seq), [0, 1, 2, 3]);
  assert.deepEqual(out.calls.map((c) => c.url.split("since=")[1]), ["0", "2", "2"]);
  assert.deepEqual(out.statuses, ["open", "retrying", "retrying", "open"]);
  assert.ok(warnings.some((w) => w.includes("reconnecting from 2")));
});

test("a stream that ends without its terminal event reconnects", async () => {
  const out = await run([
    { status: 200, chunks: [frame(0, "text", { text: "a" })] },
    { status: 200, chunks: [frame(1, "error", { status: "stopped", kind: "stopped", message: "the turn was stopped" })] },
  ]);
  assert.deepEqual(out.seen.map((e) => e.type), ["text", "error"]);
  assert.equal(out.calls[1].url.endsWith("since=1"), true);
});

test("401 and 404 are fatal; endless drops give up", async () => {
  const gone = await run([{ status: 404, chunks: [], body: { error: "no such turn" } }]);
  assert.equal(gone.fatal.status, 404);
  assert.equal(gone.fatal.message, "no such turn");
  const locked = await run([{ status: 401, chunks: [], body: { error: "bad token" } }]);
  assert.equal(locked.fatal.status, 401);
  const flaky = await run(Array.from({ length: 5 }, () => ({ throws: "offline" })), { maxRetries: 3 });
  assert.match(flaky.fatal.message, /lost the connection/);
  assert.equal(flaky.calls.length, 4);
});

test("close stops reading and never reconnects", async () => {
  const fake = fakeFetch([{ status: 200, chunks: [frame(0, "text", { text: "a" })], error: "aborted" }, { status: 200, chunks: [] }]);
  const seen = [];
  let stream;
  await new Promise((resolve) => {
    stream = AA.chatApi.streamTurn("turn_2", 0, { onEvent: (e) => { seen.push(e); stream.close(); resolve(); }, onFatal: () => assert.fail("not fatal") }, Object.assign({ fetch: fake.fetch }, noTimers));
  });
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(fake.calls.length, 1);
  assert.deepEqual(seen.map((e) => e.seq), [0]);
});

test("JSON calls send the CSRF and token headers and surface the server's error", async () => {
  const sent = [];
  globalThis.fetch = async (url, options) => {
    sent.push({ url, options });
    return { ok: false, status: 409, json: async () => ({ error: "a turn is already running in this session" }) };
  };
  await assert.rejects(AA.chatApi.startTurn("ses_1", { text: "hi" }), (err) => err.status === 409 && /already running/.test(err.message));
  const { url, options } = sent[0];
  assert.equal(url, "api/chat/sessions/ses_1/turns");
  assert.equal(options.method, "POST");
  assert.equal(options.headers["X-AA-CSRF"], "csrf-1");
  assert.equal(options.headers["Content-Type"], "application/json");
  assert.equal(JSON.parse(options.body).text, "hi");
  globalThis.fetch = async () => { throw new Error("refused"); };
  await assert.rejects(AA.chatApi.sessions(), (err) => err.status === 0 && /not reachable/.test(err.message));
});

test("a handler that throws is logged and the stream goes on, the event's place kept", async () => {
  const errors = [];
  const before = console.error;
  console.error = (...args) => errors.push(args.join(" "));
  const seen = [];
  const fake = fakeFetch([{ status: 200, chunks: [frame(0, "approval", { approval_id: "a" }), frame(1, "text", { text: "x" }), frame(2, "done", { status: "done" })] }]);
  await new Promise((resolve) => {
    AA.chatApi.streamTurn("turn_3", 0, {
      onEvent: (ev) => {
        seen.push(ev.seq);
        if (ev.seq === 0) throw new Error("draw failed");
        if (ev.type === "done") resolve();
      },
      onFatal: () => assert.fail("not fatal"),
    }, Object.assign({ fetch: fake.fetch }, instant));
  });
  console.error = before;
  assert.deepEqual(seen, [0, 1, 2]);
  assert.equal(fake.calls.length, 1, "no reconnect, so nothing was skipped");
  assert.ok(errors.some((e) => e.includes("approval event 0 of turn_3")));
});

test("a stream silent past the idle limit (not even a ping) is dropped and reopened", async () => {
  const opened = [];
  const fetch = async (url, options) => {
    opened.push(url);
    if (opened.length === 2) {
      const bytes = [new TextEncoder().encode(frame(0, "done", { status: "done" }))];
      return { ok: true, status: 200, body: { getReader: () => ({ read: async () => (bytes.length ? { done: false, value: bytes.shift() } : { done: true }), cancel: async () => {} }) } };
    }
    return { ok: true, status: 200, body: { getReader: () => ({
      read: () => new Promise((_, reject) => options.signal.addEventListener("abort", () => reject(new Error("aborted")))),
      cancel: async () => {},
    }) } };
  };
  const seen = [];
  await new Promise((resolve) => {
    AA.chatApi.streamTurn("turn_4", 0, { onEvent: (ev) => { seen.push(ev.type); resolve(); }, onFatal: () => assert.fail("not fatal") },
      { fetch, idleMs: 20, setTimeout: (fn, ms) => setTimeout(fn, ms), clearTimeout: (t) => clearTimeout(t) });
  });
  assert.equal(opened.length, 2);
  assert.deepEqual(seen, ["done"]);
});

test("an upload sends the bytes with the CSRF, token and encoded file name headers", async () => {
  const sent = [];
  globalThis.fetch = async (url, options) => {
    sent.push({ url, options });
    return { ok: true, status: 201, json: async () => ({ attachment_id: "fil_1", name: "ä b.txt", size: 3, media_type: "text/plain" }) };
  };
  const file = { name: "ä b.txt", size: 3, type: "text/plain" };
  const ref = await AA.chatApi.upload("ses_1", file);
  assert.equal(ref.attachment_id, "fil_1");
  const { url, options } = sent[0];
  assert.equal(url, "api/chat/files?session=ses_1");
  assert.equal(options.method, "POST");
  assert.equal(options.body, file);
  assert.equal(options.headers["Content-Type"], "application/octet-stream");
  assert.equal(options.headers["X-AA-CSRF"], "csrf-1");
  assert.equal(options.headers["X-AA-Token"], "tok/en");
  assert.equal(options.headers["X-AA-Filename"], "%C3%A4%20b.txt");
});

test("the token comes only from data.js", () => {
  assert.equal(AA.chatApi.token, undefined);
  const source = readFileSync(join(STATIC, "chatapi.js"), "utf8");
  assert.ok(!source.includes("sessionStorage") && !source.includes("location.hash"));
  assert.ok(source.includes("AA.token()"));
});
