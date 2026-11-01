import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { ForbiddenError, ToolProtocolError, ToolServerError, UnauthorizedError } from "../src/index.ts";
import { http, problem, rejection } from "./helpers.ts";

describe("HttpClient response size", () => {
  test("a success body larger than maxResponseSize is a protocol error", async () => {
    const big = new Response(JSON.stringify({ data: "x".repeat(100) }), { status: 200 });
    const { client } = http([big], { maxResponseSize: 50 });
    const err = await rejection(client.request({ method: "GET", path: "/a" }));
    assert.ok(err instanceof ToolProtocolError);
    assert.match(err.message, /exceeds 50 bytes/);
  });

  test("a declared Content-Length over the limit is refused before reading", async () => {
    let cancelled = false;
    const body = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("{}")); // small: only the header can trip the limit
      },
      cancel() {
        cancelled = true;
      },
    });
    const res = new Response(body, { status: 200, headers: { "content-length": "1000" } });
    const { client } = http([res], { maxResponseSize: 50 });
    await assert.rejects(client.request({ method: "GET", path: "/a" }), /exceeds 50 bytes/);
    assert.ok(cancelled);
  });

  test("an error body larger than maxResponseSize is a protocol error", async () => {
    const { client } = http([problem(404, "tool-not-found", { pad: "x".repeat(200) })], { maxResponseSize: 50 });
    await assert.rejects(client.request({ method: "GET", path: "/a" }), ToolProtocolError);
  });

  test("an oversized body whose cancel fails still raises the size error", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(100));
      },
      cancel() {
        throw new Error("cannot cancel");
      },
    });
    const { client } = http([new Response(body, { status: 200 })], { maxResponseSize: 50 });
    await assert.rejects(client.request({ method: "GET", path: "/a" }), /exceeds 50 bytes/);
  });

  test("a body exactly at the limit is read", async () => {
    const text = JSON.stringify({ a: "x".repeat(10) });
    const { client } = http([new Response(text, { status: 200 })], { maxResponseSize: text.length });
    assert.deepEqual((await client.request({ method: "GET", path: "/a" })).body, { a: "x".repeat(10) });
  });
});

describe("problem bodies", () => {
  const problemResponse = (body: string, status: number, headers: Record<string, string> = {}) =>
    new Response(body, { status, headers: { "content-type": "application/problem+json; charset=utf-8", ...headers } });

  test("a problem without a numeric status takes the HTTP status", async () => {
    const body = JSON.stringify({ type: "urn:agent-adapter:error:forbidden", title: "Forbidden" });
    const { client } = http([problemResponse(body, 403)]);
    const err = await rejection(client.request({ method: "GET", path: "/a" }));
    assert.ok(err instanceof ForbiddenError);
    assert.equal(err.problem.status, 403);
  });

  test("a problem body that is not an object, or lacks type or title, becomes about:blank", async () => {
    for (const body of ["[1]", "null", '{"type":"x"}', '{"title":"x"}']) {
      const { client } = http([problemResponse(body, 404)]);
      const err = await rejection(client.request({ method: "GET", path: "/a" }));
      assert.ok(err instanceof ToolServerError, body);
      assert.equal(err.problem.type, "about:blank", body);
    }
  });

  test("an error without a body or content type still maps by status", async () => {
    const { client } = http([new Response(null, { status: 401 })]);
    const err = await rejection(client.request({ method: "GET", path: "/a" }));
    assert.ok(err instanceof UnauthorizedError);
    assert.equal(err.problem.title, "HTTP 401");
  });
});
