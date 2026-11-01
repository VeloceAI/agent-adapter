import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { asVercelTools, ToolClient, ToolFailedError, toolDefinitions } from "../src/index.ts";
import { fakeFetch, json, problem } from "./helpers.ts";
import type { Step } from "./helpers.ts";

function descriptor(name: string) {
  return {
    name,
    description: `${name} tool`,
    input_schema: { type: "object", properties: {} },
    output_schema: null,
    kind: "",
    tags: [],
    effects: null,
    idempotent: null,
    approval: "never",
    requires_approval: false,
    parallel_safe: true,
    timeout: null,
    retries: 0,
    cache_ttl: null,
    prompt_hint: null,
  };
}
const catalogue = () => json(200, { version: "1", tools: [descriptor("add"), descriptor("send")] });
const OK = { status: "ok", run_id: "r", call_id: "c", result: { sum: 3 }, cached: false, attempts: 1 };
const PENDING = { status: "pending_approval", run_id: "r", call_id: "c", approval_id: "apr_9" };

function client(steps: Step[]) {
  const fake = fakeFetch(...steps);
  return { tc: new ToolClient({ baseUrl: "http://srv", fetch: fake.fetch, retry: false }), fake };
}

describe("toolDefinitions", () => {
  test("maps the catalogue and filters by tags", async () => {
    const { tc, fake } = client([catalogue()]);
    const defs = await tc.toolDefinitions({ tags: ["x"] });
    assert.deepEqual(
      defs.map((d) => [d.name, d.description, d.inputSchema]),
      [
        ["add", "add tool", { type: "object", properties: {} }],
        ["send", "send tool", { type: "object", properties: {} }],
      ],
    );
    assert.equal(fake.requests[0]!.url.searchParams.get("tags"), "x");
  });

  test("execute returns the result, or the approval id while pending", async () => {
    const { tc, fake } = client([catalogue(), json(200, OK), json(202, PENDING)]);
    const [add, send] = await toolDefinitions(tc, { runId: "run_7", threadId: "th" });
    assert.deepEqual(await add!.execute({ a: 1 }), { sum: 3 });
    assert.deepEqual(await send!.execute({ to: "x" }), { pending_approval: "apr_9" });
    assert.deepEqual(JSON.parse(fake.requests[1]!.body!), { arguments: { a: 1 }, run_id: "run_7", thread_id: "th" });
    assert.equal(fake.requests[2]!.url.pathname, "/tools/send/invoke");
  });

  test("execute forwards the host's abort signal", async () => {
    const { tc } = client([catalogue()]);
    const [add] = await tc.toolDefinitions();
    await assert.rejects(add!.execute({}, { abortSignal: AbortSignal.abort(new Error("cancelled")) }), /cancelled/);
  });

  test("execute raises server errors", async () => {
    const { tc } = client([catalogue(), problem(502, "tool-failed")]);
    const [add] = await tc.toolDefinitions();
    await assert.rejects(add!.execute({}), ToolFailedError);
  });
});

describe("asVercelTools", () => {
  test("builds one AI SDK tool per catalogue entry", async () => {
    const { tc } = client([catalogue(), json(200, OK)]);
    const schemas: unknown[] = [];
    const tool = (def: unknown) => ({ wrapped: def });
    const jsonSchema = (schema: unknown) => {
      schemas.push(schema);
      return { jsonSchema: schema };
    };
    const tools = (await tc.asVercelTools(tool, jsonSchema)) as Record<
      string,
      { wrapped: { description: string; inputSchema: unknown; execute: (i: object) => Promise<unknown> } }
    >;
    assert.deepEqual(Object.keys(tools), ["add", "send"]);
    assert.equal(tools["add"]!.wrapped.description, "add tool");
    assert.deepEqual(tools["add"]!.wrapped.inputSchema, { jsonSchema: { type: "object", properties: {} } });
    assert.equal(schemas.length, 2);
    assert.deepEqual(await tools["add"]!.wrapped.execute({}), { sum: 3 });
  });

  test("a tool named __proto__ becomes an own key", async () => {
    const { tc } = client([json(200, { version: "1", tools: [descriptor("__proto__")] })]);
    const tools = await asVercelTools(
      tc,
      (def: unknown) => def,
      (schema: unknown) => schema,
    );
    assert.ok(Object.hasOwn(tools, "__proto__"));
    assert.equal(Object.getPrototypeOf(tools), Object.prototype);
  });
});
