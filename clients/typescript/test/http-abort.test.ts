import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { ToolTransportError } from "../src/index.ts";
import { hang, http, json, rejection } from "./helpers.ts";

describe("HttpClient timeouts and aborts", () => {
  test("a timed-out attempt is a timeout transport error", async () => {
    const { client } = http([hang], { timeoutMs: 20, retry: false });
    const err = await rejection(client.request({ method: "GET", path: "/a" }));
    assert.ok(err instanceof ToolTransportError);
    assert.equal(err.reason, "timeout");
  });

  test("GET retries after a timeout", async () => {
    const { client, fake } = http([hang, json(200, { ok: true })], { timeoutMs: 20 });
    assert.deepEqual((await client.request({ method: "GET", path: "/a" })).body, { ok: true });
    assert.equal(fake.requests.length, 2);
  });

  test("aborting while an async token provider runs sends nothing", async () => {
    const controller = new AbortController();
    const token = () => new Promise<string>((resolve) => setTimeout(() => resolve("t"), 40));
    const { client, fake } = http([json(200, {})], { token });
    setTimeout(() => controller.abort(new Error("cancelled during auth")), 5);
    const err = await rejection(
      client.request({ method: "POST", path: "/invoke", body: {}, idempotent: true, signal: controller.signal }),
    );
    assert.equal(err.message, "cancelled during auth");
    await new Promise((resolve) => setTimeout(resolve, 60));
    assert.equal(fake.requests.length, 0);
  });

  test("a failing token provider fails the request as is", async () => {
    const failure = new Error("vault down");
    const { client, fake } = http([json(200, {})], { token: () => Promise.reject(failure) });
    const err = await rejection(client.request({ method: "GET", path: "/a", signal: new AbortController().signal }));
    assert.equal(err, failure);
    assert.equal(fake.requests.length, 0);
  });

  test("an abort that lands as the token resolves sends nothing", async () => {
    const controller = new AbortController();
    const token = () => {
      controller.abort(new Error("late abort"));
      return "t";
    };
    const { client, fake } = http([json(200, {})], { token });
    const err = await rejection(
      client.request({ method: "POST", path: "/invoke", body: {}, idempotent: true, signal: controller.signal }),
    );
    assert.equal(err.message, "late abort");
    assert.equal(fake.requests.length, 0);
  });

  test("a POST is not retried after a timeout", async () => {
    const { client, fake } = http([hang, json(200, {})], { timeoutMs: 20 });
    const err = await rejection(client.request({ method: "POST", path: "/a", body: {}, idempotent: true }));
    assert.ok(err instanceof ToolTransportError);
    assert.equal(fake.requests.length, 1);
  });

  test("an already aborted signal rejects with its reason and sends nothing", async () => {
    const { client, fake } = http([json(200, {})]);
    const reason = new Error("stop");
    const err = await rejection(client.request({ method: "GET", path: "/a", signal: AbortSignal.abort(reason) }));
    assert.equal(err, reason);
    assert.equal(fake.requests.length, 0);
  });

  test("aborting during a request rejects with the reason and does not retry", async () => {
    const controller = new AbortController();
    const { client, fake } = http([hang, json(200, {})]);
    const pending = client.request({ method: "GET", path: "/a", signal: controller.signal });
    setTimeout(() => controller.abort(new Error("user stop")), 5);
    const err = await rejection(pending);
    assert.equal(err.message, "user stop");
    assert.equal(fake.requests.length, 1);
  });

  test("aborting during a retry wait rejects promptly", async () => {
    const controller = new AbortController();
    const { client } = http([json(503, {}, { "retry-after": "5" })], { retry: { maxDelayMs: 10_000 } });
    const started = performance.now();
    const pending = client.request({ method: "GET", path: "/a", signal: controller.signal });
    setTimeout(() => controller.abort(new Error("bye")), 20);
    const err = await rejection(pending);
    assert.equal(err.message, "bye");
    assert.ok(performance.now() - started < 2000);
  });
});
