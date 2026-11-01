import assert from "node:assert/strict";
import { describe, test } from "node:test";
import type { TestContext } from "node:test";

import { add, approveAndWatch, connect, runCost, vercelTools } from "../examples/quickstart.ts";
import { json, problem, sse } from "./helpers.ts";

type Route = (body: unknown) => Response;

/** Replaces the global fetch with a router keyed by "METHOD /path" for the test's duration. */
function serve(t: TestContext, routes: Record<string, Route>): Array<{ route: string; body: unknown }> {
  const seen: Array<{ route: string; body: unknown }> = [];
  t.mock.method(globalThis, "fetch", async (input: string | URL, init: RequestInit = {}) => {
    const url = new URL(String(input));
    const route = `${init.method ?? "GET"} ${url.pathname}`;
    const body = typeof init.body === "string" ? JSON.parse(init.body) : undefined;
    seen.push({ route, body });
    const handler = routes[route];
    assert.ok(handler, `no route for ${route}`);
    return handler(body);
  });
  return seen;
}

const DESCRIPTOR = {
  name: "add",
  description: "Add two numbers.",
  input_schema: { type: "object", properties: { a: { type: "number" }, b: { type: "number" } } },
  output_schema: null,
  kind: "",
  tags: ["math"],
  effects: null,
  idempotent: true,
  approval: "never",
  requires_approval: false,
  parallel_safe: true,
  timeout: null,
  retries: 0,
  cache_ttl: null,
  prompt_hint: null,
};
const CATALOGUE = () => json(200, { version: "1", tools: [DESCRIPTOR] });
const OK = (result: unknown) =>
  json(200, { status: "ok", run_id: "run_1", call_id: "call_1", result, cached: false, attempts: 1 });

describe("examples/quickstart.ts", () => {
  test("connect lists the math tools", async (t) => {
    const seen = serve(t, { "GET /api/tools": CATALOGUE });
    const { tools } = await connect("https://tools.internal/api", "token");
    assert.equal(tools[0]?.name, "add");
    assert.equal(seen[0]?.route, "GET /api/tools");
  });

  test("add returns the sum, or the schema errors", async (t) => {
    let invalid = false;
    serve(t, {
      "GET /tools": CATALOGUE,
      "POST /tools/add/invoke": () =>
        invalid ? problem(422, "tool-args-invalid", { errors: [{ loc: ["b"] }] }) : OK({ sum: 5 }),
    });
    const { client } = await connect("https://tools.internal", "token");
    assert.equal(await add(client, "run_1"), 5);
    invalid = true;
    assert.deepEqual(await add(client, "run_1"), [{ loc: ["b"] }]);
  });

  test("approveAndWatch edits, approves and sees the events", async (t) => {
    const frames = [
      'id: 3\nevent: approval.decided\ndata: {"type":"approval.decided"}\n\n',
      'id: 4\nevent: tool.finished\ndata: {"type":"tool.finished"}\n\n',
      "event: end\ndata: {}\n\n",
    ];
    const seen = serve(t, {
      "GET /tools": CATALOGUE,
      "POST /tools/send_email/invoke": () =>
        json(202, { status: "pending_approval", run_id: "run_1", call_id: "call_1", approval_id: "apr_1" }),
      "GET /approvals/apr_1": () =>
        json(200, {
          approval_id: "apr_1",
          status: "pending",
          tool: "send_email",
          arguments: { to: "ops@example.com" },
        }),
      "POST /approvals/apr_1/decide": () => OK({ sent: true }),
      "GET /runs/run_1/events": () => sse(frames),
    });
    const { client } = await connect("https://tools.internal", "token");
    const outcome = await approveAndWatch(client, "run_1");
    assert.deepEqual(outcome, { result: { sent: true }, events: ["approval.decided", "tool.finished"] });
    const decide = seen.find((s) => s.route === "POST /approvals/apr_1/decide");
    assert.deepEqual(decide?.body, {
      approved: true,
      edited_arguments: { to: "ops@example.com", cc: "audit@example.com" },
      reason: "checked by ops",
    });
  });

  test("runCost reads the total", async (t) => {
    serve(t, {
      "GET /tools": CATALOGUE,
      "GET /runs/run_1/cost": () => json(200, { run_id: "run_1", total_usd: 0.0123 }),
    });
    const { client } = await connect("https://tools.internal", "token");
    assert.equal(await runCost(client, "run_1"), 0.0123);
  });

  test("vercelTools wraps every tool", async (t) => {
    serve(t, { "GET /tools": CATALOGUE });
    const { client } = await connect("https://tools.internal", "token");
    const tools = await vercelTools(client, { tool: (def) => ({ def }), jsonSchema: (schema) => schema }, "run_1");
    assert.deepEqual(Object.keys(tools), ["add"]);
  });
});
