import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  BudgetExceededError,
  NotFoundError,
  ToolProtocolError,
  ToolServerError,
  ToolTransportError,
  UnauthorizedError,
} from "../src/index.ts";
import { HttpClient, newIdempotencyKey, parseRetryAfter, sleep } from "../src/http.ts";
import { FAST_RETRY, http, json, networkError, problem, rejection } from "./helpers.ts";

const TOKEN = "s3cr3t-token";

describe("HttpClient requests", () => {
  test("joins the base URL and path and drops undefined query values", async () => {
    const { client, fake } = http([json(200, { ok: 1 })]);
    const res = await client.request({ method: "GET", path: "/tools", query: { shape: "openai", tags: undefined } });
    assert.deepEqual(res.body, { ok: 1 });
    assert.equal(res.status, 200);
    assert.equal(fake.requests[0]?.url.href, "http://srv/api/tools?shape=openai");
  });

  test("sends accept, user headers and a bearer token", async () => {
    const { client, fake } = http([json(200, {})], { token: TOKEN, headers: { "x-team": "a" } });
    await client.request({ method: "GET", path: "/tools" });
    const headers = fake.requests[0]!.headers;
    assert.equal(headers.get("authorization"), `Bearer ${TOKEN}`);
    assert.equal(headers.get("x-team"), "a");
    assert.match(headers.get("accept") ?? "", /application\/json/);
    assert.match(headers.get("accept") ?? "", /application\/problem\+json/);
  });

  test("the token wins over a user Authorization header", async () => {
    const { client, fake } = http([json(200, {})], { token: TOKEN, headers: { Authorization: "Basic x" } });
    await client.request({ method: "GET", path: "/tools" });
    assert.equal(fake.requests[0]!.headers.get("authorization"), `Bearer ${TOKEN}`);
  });

  test("omits Authorization without a token or with an empty one", async () => {
    const { client, fake } = http([json(200, {}), json(200, {})], { token: () => "" });
    await client.request({ method: "GET", path: "/a" });
    assert.equal(fake.requests[0]!.headers.has("authorization"), false);
    const other = http([json(200, {})]);
    await other.client.request({ method: "GET", path: "/a" });
    assert.equal(other.fake.requests[0]!.headers.has("authorization"), false);
  });

  test("resolves an async token before every attempt", async () => {
    let calls = 0;
    const token = async () => `t${++calls}`;
    const { client, fake } = http([json(503, {}, { "retry-after": "0" }), json(200, {})], { token });
    await client.request({ method: "GET", path: "/a" });
    assert.deepEqual(
      fake.requests.map((r) => r.headers.get("authorization")),
      ["Bearer t1", "Bearer t2"],
    );
  });

  test("serialises a POST body as JSON", async () => {
    const { client, fake } = http([json(200, {})]);
    await client.request({ method: "POST", path: "/x", body: { a: 1 }, headers: { "idempotency-key": "k" } });
    const req = fake.requests[0]!;
    assert.equal(req.method, "POST");
    assert.equal(req.headers.get("content-type"), "application/json");
    assert.equal(req.headers.get("idempotency-key"), "k");
    assert.equal(req.body, '{"a":1}');
  });

  test("an invalid JSON success body is a protocol error", async () => {
    const { client } = http([new Response("not json", { status: 200 })]);
    const err = await rejection(client.request({ method: "GET", path: "/a" }));
    assert.ok(err instanceof ToolProtocolError);
    assert.equal(err.status, 200);
  });

  test("a problem body becomes the typed error with Retry-After", async () => {
    const { client } = http([problem(429, "budget-exceeded", {}, { "retry-after": "60" })], { retry: false });
    const err = await rejection(client.request({ method: "POST", path: "/x", body: {}, idempotent: true }));
    assert.ok(err instanceof BudgetExceededError);
    assert.equal(err.retryAfterMs, 60_000);
    assert.equal(err.problem.detail, "detail of budget-exceeded");
  });

  test("a non-problem error body becomes an about:blank problem", async () => {
    const res = new Response("<html>gateway</html>", {
      status: 404,
      statusText: "Not Found",
      headers: { "content-type": "text/html" },
    });
    const { client } = http([res]);
    const err = await rejection(client.request({ method: "GET", path: "/a" }));
    assert.ok(err instanceof NotFoundError);
    assert.equal(err.problem.type, "about:blank");
    assert.equal(err.problem.title, "Not Found");
    assert.equal(err.problem.status, 404);
  });

  test("a malformed problem body falls back to about:blank", async () => {
    const res = new Response("{oops", { status: 401, headers: { "content-type": "application/problem+json" } });
    const { client } = http([res]);
    const err = await rejection(client.request({ method: "GET", path: "/a" }));
    assert.ok(err instanceof UnauthorizedError);
    assert.equal(err.problem.type, "about:blank");
    assert.equal(err.problem.title, "HTTP 401");
  });

  test("errors never carry the token", async () => {
    const { client } = http([problem(401, "unauthorized"), networkError("ECONNRESET")], { token: TOKEN, retry: false });
    for (const promise of [
      client.request({ method: "GET", path: "/a" }),
      client.request({ method: "GET", path: "/b" }),
    ]) {
      const err = await rejection(promise);
      const text = `${err.message} ${String(err)} ${JSON.stringify(err)} ${JSON.stringify((err as ToolServerError).problem ?? {})}`;
      assert.ok(!text.includes(TOKEN));
    }
  });

  test("rejects a bad base URL", () => {
    for (const baseUrl of ["not a url", "http://user:pw@h/", "http://h/?a=1", "http://h/#x", "ftp://h/"]) {
      assert.throws(() => new HttpClient({ baseUrl }), TypeError, baseUrl);
    }
  });

  test("rejects bad numeric options", () => {
    assert.throws(() => new HttpClient({ baseUrl: "http://h", timeoutMs: 0 }), RangeError);
    assert.throws(() => new HttpClient({ baseUrl: "http://h", retry: { maxRetries: -1 } }), RangeError);
    assert.throws(() => new HttpClient({ baseUrl: "http://h", retry: { baseDelayMs: Number.NaN } }), RangeError);
    assert.throws(() => new HttpClient({ baseUrl: "http://h", maxResponseSize: 0 }), RangeError);
  });

  test("an invalid base URL error does not echo the URL", () => {
    const err = (() => {
      try {
        new HttpClient({ baseUrl: "http://user:s3cret@bad host/" });
      } catch (caught) {
        return caught as TypeError;
      }
      return assert.fail("expected a TypeError");
    })();
    assert.ok(err instanceof TypeError);
    const text = `${err.message} ${String(err)} ${JSON.stringify(err)} ${JSON.stringify(err.cause ?? null)}`;
    assert.ok(!text.includes("s3cret"), text);
  });

  test("a token with control characters is refused without echoing it", async () => {
    for (const token of ["abc\r\nX-Evil: 1", "abc\u0000", () => "abc\n"]) {
      const { client, fake } = http([json(200, {})], { token });
      const err = await rejection(client.request({ method: "GET", path: "/a" }));
      assert.ok(err instanceof TypeError);
      assert.equal(err.message, "invalid token characters");
      assert.ok(!JSON.stringify(err).includes("abc"));
      assert.equal(fake.requests.length, 0);
    }
  });

  test("a custom fetch is called without this (browser fetch needs that)", async () => {
    let calls = 0;
    const strictFetch = function (this: unknown, _input: unknown, _init: unknown): Promise<Response> {
      calls++;
      if (this !== undefined) throw new TypeError("Illegal invocation");
      return Promise.resolve(json(200, { ok: 1 }));
    };
    const client = new HttpClient({ baseUrl: "http://h", fetch: strictFetch as typeof fetch, retry: FAST_RETRY });
    assert.deepEqual((await client.request({ method: "GET", path: "/a" })).body, { ok: 1 });
    assert.equal(calls, 1);
  });

  test("a fetch that throws while setting up the call is not a retryable network error", async () => {
    let calls = 0;
    const setupError = new TypeError("Request cannot be constructed");
    const throwing = (): Promise<Response> => {
      calls++;
      throw setupError;
    };
    const client = new HttpClient({ baseUrl: "http://h", fetch: throwing as typeof fetch, retry: FAST_RETRY });
    const err = await rejection(client.request({ method: "GET", path: "/a" }));
    assert.equal(err, setupError);
    assert.equal(calls, 1);
  });
});

describe("HttpClient retry policy", () => {
  test("GET retries network errors and 408/429/502/503/504", async () => {
    for (const first of [
      networkError("ECONNRESET"),
      json(408, {}),
      json(429, {}),
      json(502, {}),
      json(503, {}),
      json(504, {}),
    ]) {
      const { client, fake } = http([first, json(200, { ok: 1 })]);
      assert.deepEqual((await client.request({ method: "GET", path: "/a" })).body, { ok: 1 });
      assert.equal(fake.requests.length, 2);
    }
  });

  test("GET does not retry other statuses", async () => {
    const { client, fake } = http([problem(404, "tool-not-found"), json(200, {})]);
    await rejection(client.request({ method: "GET", path: "/a" }));
    assert.equal(fake.requests.length, 1);
  });

  test("retries are bounded and the last error is thrown", async () => {
    const { client, fake } = http([json(503, {}), json(503, {}), problem(503, "internal-error")]);
    const err = await rejection(client.request({ method: "GET", path: "/a" }));
    assert.ok(err instanceof ToolServerError);
    assert.equal(err.kind, "internal-error");
    assert.equal(fake.requests.length, 3);
  });

  test("retry: false sends one request", async () => {
    const { client, fake } = http([json(503, {}), json(200, {})], { retry: false });
    await rejection(client.request({ method: "GET", path: "/a" }));
    assert.equal(fake.requests.length, 1);
  });

  test("a Retry-After beyond maxDelayMs is not waited for", async () => {
    const { client, fake } = http([json(503, {}, { "retry-after": "120" }), json(200, {})]);
    const err = await rejection(client.request({ method: "GET", path: "/a" }));
    assert.ok(err instanceof ToolServerError);
    assert.equal(err.retryAfterMs, 120_000);
    assert.equal(fake.requests.length, 1);
  });

  test("an idempotent POST retries connect failures and 429/503 with Retry-After", async () => {
    for (const first of [
      networkError("ECONNREFUSED"),
      networkError("ENOTFOUND"),
      networkError("UND_ERR_CONNECT_TIMEOUT"),
      json(503, {}, { "retry-after": "0" }),
      json(429, {}, { "retry-after": "0" }),
    ]) {
      const { client, fake } = http([first, json(200, { ok: 1 })]);
      const res = await client.request({ method: "POST", path: "/a", body: {}, idempotent: true });
      assert.deepEqual(res.body, { ok: 1 });
      assert.equal(fake.requests.length, 2);
    }
  });

  test("an idempotent POST does not retry when the server may have run the tool", async () => {
    for (const first of [
      networkError("ECONNRESET"),
      new TypeError("Failed to fetch"),
      json(503, {}),
      json(429, {}),
      problem(502, "tool-failed"),
      problem(504, "tool-timeout"),
      json(500, {}),
    ]) {
      const { client, fake } = http([first, json(200, {})]);
      await rejection(client.request({ method: "POST", path: "/a", body: {}, idempotent: true }));
      assert.equal(fake.requests.length, 1);
    }
  });

  test("a POST without idempotency is never retried", async () => {
    const { client, fake } = http([networkError("ECONNREFUSED"), json(200, {})]);
    const err = await rejection(client.request({ method: "POST", path: "/a", body: {} }));
    assert.ok(err instanceof ToolTransportError);
    assert.equal(err.reason, "network");
    assert.ok(err.cause instanceof TypeError);
    assert.equal(fake.requests.length, 1);
  });

  test("a retried response whose body already failed is still discarded", async () => {
    const body = new ReadableStream({
      start(controller) {
        controller.error(new Error("broken body"));
      },
    });
    const { client } = http([new Response(body, { status: 503 }), json(200, { ok: 1 })]);
    assert.deepEqual((await client.request({ method: "GET", path: "/a" })).body, { ok: 1 });
  });

  // What each runtime's fetch rejects with when the network fails; none carries a code.
  const UNCODED_FAILURES = [
    "Failed to fetch", // Chrome
    "NetworkError when attempting to fetch resource.", // Firefox
    "Load failed", // Safari
    "error sending request for url (http://srv/api/a): Connection refused (os error 111)", // Deno
    "Network request failed", // React Native, whatwg-fetch
    "Network connection lost.", // Cloudflare Workers
    "Request cannot be constructed from a URL that includes credentials", // a rejected setup error
  ];

  test("every rejected fetch is a network error, with no code when the runtime gives none", async () => {
    for (const message of UNCODED_FAILURES) {
      const { client } = http([new TypeError(message)], { retry: false });
      const err = await rejection(client.request({ method: "GET", path: "/a" }));
      assert.ok(err instanceof ToolTransportError, message);
      assert.equal(err.reason, "network");
      assert.equal(err.code, "");
    }
  });

  test("GETs retry uncoded network failures", async () => {
    for (const message of UNCODED_FAILURES) {
      const { client, fake } = http([new TypeError(message), json(200, { ok: 1 })]);
      assert.deepEqual((await client.request({ method: "GET", path: "/a" })).body, { ok: 1 }, message);
      assert.equal(fake.requests.length, 2);
    }
  });

  test("POSTs never retry uncoded network failures", async () => {
    for (const message of UNCODED_FAILURES) {
      const { client, fake } = http([new TypeError(message), json(200, {})]);
      await assert.rejects(
        client.request({ method: "POST", path: "/a", body: {}, idempotent: true }),
        ToolTransportError,
      );
      assert.equal(fake.requests.length, 1, message);
    }
  });

  test("a coded error from another fetch implementation is a network error", async () => {
    const hangUp = Object.assign(new Error("socket hang up"), { code: "ECONNRESET" });
    const { client, fake } = http([hangUp, json(200, {})]);
    await client.request({ method: "GET", path: "/a" });
    assert.equal(fake.requests.length, 2);
  });

  test("a refused connection reported as an AggregateError is a connect failure", async () => {
    const refused = Object.assign(new Error("connect ECONNREFUSED ::1:80"), { code: "ECONNREFUSED" });
    const failure = new TypeError("fetch failed", { cause: new AggregateError([refused]) });
    const { client, fake } = http([failure, json(200, {})]);
    await client.request({ method: "POST", path: "/a", body: {}, idempotent: true });
    assert.equal(fake.requests.length, 2);
  });

  test("the body of a retried response is released", async () => {
    let cancelled = false;
    const body = new ReadableStream({
      cancel() {
        cancelled = true;
      },
    });
    const { client } = http([new Response(body, { status: 503 }), json(200, {})]);
    await client.request({ method: "GET", path: "/a" });
    assert.ok(cancelled);
  });
});

describe("HttpClient streams", () => {
  test("the timeout covers the headers only", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        setTimeout(() => {
          controller.enqueue(new TextEncoder().encode("late"));
          controller.close();
        }, 60);
      },
    });
    const { client } = http([new Response(body, { status: 200 })], { timeoutMs: 20 });
    const stream = await client.open({ method: "GET", path: "/events" });
    assert.equal(await stream.response.text(), "late");
    stream.close();
  });

  test("close aborts the request", async () => {
    const { client, fake } = http([new Response(new ReadableStream(), { status: 200 })]);
    const stream = await client.open({ method: "GET", path: "/events" });
    stream.close();
    assert.equal(fake.requests[0]!.signal?.aborted, true);
  });

  test("an error status throws the typed error", async () => {
    const { client } = http([problem(403, "forbidden")]);
    const err = await rejection(client.open({ method: "GET", path: "/events" }));
    assert.equal(err.name, "ForbiddenError");
  });
});

describe("helpers", () => {
  test("parseRetryAfter reads seconds and HTTP dates", () => {
    const now = Date.parse("2026-10-30T00:00:00Z");
    assert.equal(parseRetryAfter("2", now), 2000);
    assert.equal(parseRetryAfter("0", now), 0);
    assert.equal(parseRetryAfter("Fri, 30 Oct 2026 00:00:05 GMT", now), 5000);
    assert.equal(parseRetryAfter("Thu, 29 Oct 2026 00:00:00 GMT", now), 0);
    for (const bad of [null, "", "-1", "1.5", "soon"]) assert.equal(parseRetryAfter(bad, now), undefined);
  });

  test("sleep rejects at once on an aborted signal", async () => {
    await assert.rejects(sleep(10_000, AbortSignal.abort(new Error("gone"))), /gone/);
  });

  test("newIdempotencyKey gives distinct v4 UUIDs, also without randomUUID", () => {
    const pattern = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
    const a = newIdempotencyKey();
    assert.match(a, pattern);
    assert.notEqual(a, newIdempotencyKey());
    const bytesOnly = { getRandomValues: crypto.getRandomValues.bind(crypto) };
    assert.match(newIdempotencyKey(bytesOnly), pattern);
  });
});
