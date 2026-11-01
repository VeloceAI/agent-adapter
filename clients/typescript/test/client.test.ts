import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { inspect } from "node:util";

import {
  ApprovalAlreadyDecidedError,
  NotFoundError,
  ToolArgsInvalidError,
  ToolClient,
  ToolNotFoundError,
  ToolProtocolError,
} from "../src/index.ts";
import type { ClientOptions } from "../src/index.ts";
import { FAST_RETRY, fakeFetch, json, networkError, problem } from "./helpers.ts";
import type { Step } from "./helpers.ts";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const TOOL = {
  name: "add",
  description: "Add two numbers.",
  input_schema: { type: "object", properties: { a: { type: "number" } } },
  output_schema: null,
  kind: "compute",
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
const OK = { status: "ok", run_id: "run_1", call_id: "call_1", result: 3, cached: false, attempts: 1 };
const PENDING = { status: "pending_approval", run_id: "run_1", call_id: "call_1", approval_id: "apr_1" };
const APPROVAL = {
  approval_id: "apr_1",
  status: "pending",
  tool: "send",
  arguments: { to: "a" },
  run_id: "run_1",
  call_id: "call_1",
  hint: "",
  created_at: 1,
  decision: null,
};

function client(steps: Step[], options: Partial<ClientOptions> = {}) {
  const fake = fakeFetch(...steps);
  const tc = new ToolClient({ baseUrl: "https://srv/aa", fetch: fake.fetch, retry: FAST_RETRY, ...options });
  return { tc, fake };
}

describe("health", () => {
  test("health reads /healthz", async () => {
    const { tc, fake } = client([json(200, { status: "ok", version: "0.1.0" })]);
    assert.deepEqual(await tc.health(), { status: "ok", version: "0.1.0" });
    assert.equal(fake.requests[0]!.url.pathname, "/aa/healthz");
  });
});

describe("catalogue", () => {
  test("listTools sends shape, tags and kinds and returns the tools", async () => {
    const { tc, fake } = client([json(200, { version: "1", tools: [TOOL] })]);
    const tools = await tc.listTools({ shape: "openai", tags: ["a", "b"], kinds: ["compute"] });
    assert.deepEqual(tools, [TOOL]);
    const url = fake.requests[0]!.url;
    assert.equal(url.pathname, "/aa/tools");
    assert.equal(url.searchParams.get("shape"), "openai");
    assert.equal(url.searchParams.get("tags"), "a,b");
    assert.equal(url.searchParams.get("kinds"), "compute");
  });

  test("listTools without options sends no query", async () => {
    const { tc, fake } = client([json(200, { version: "1", tools: [] })]);
    assert.deepEqual(await tc.listTools(), []);
    assert.equal(fake.requests[0]!.url.search, "");
  });

  test("listTools rejects a catalogue without a tools list", async () => {
    const { tc } = client([json(200, { version: "1" })]);
    await assert.rejects(tc.listTools(), ToolProtocolError);
  });

  test("getTool encodes the name and passes the shape", async () => {
    const { tc, fake } = client([json(200, TOOL)]);
    assert.deepEqual(await tc.getTool("a/b c", { shape: "gemini" }), TOOL);
    assert.equal(fake.requests[0]!.url.pathname, "/aa/tools/a%2Fb%20c");
    assert.equal(fake.requests[0]!.url.searchParams.get("shape"), "gemini");
  });

  test("getTool raises ToolNotFoundError", async () => {
    const { tc } = client([problem(404, "tool-not-found", { tool: "nope" })]);
    await assert.rejects(tc.getTool("nope"), (err: unknown) => err instanceof ToolNotFoundError && err.tool === "nope");
  });

  test("passes a caller signal through", async () => {
    const { tc } = client([json(200, { tools: [] })]);
    await assert.rejects(tc.listTools({ signal: AbortSignal.abort(new Error("x")) }), /x/);
  });
});

describe("invoke", () => {
  test("posts arguments, run and thread ids with a generated Idempotency-Key", async () => {
    const { tc, fake } = client([json(200, OK)]);
    const result = await tc.invoke<number>("add", { a: 1 }, { runId: "run_1", threadId: "t1" });
    assert.deepEqual(result, OK);
    const req = fake.requests[0]!;
    assert.equal(req.method, "POST");
    assert.equal(req.url.pathname, "/aa/tools/add/invoke");
    assert.deepEqual(JSON.parse(req.body!), { arguments: { a: 1 }, run_id: "run_1", thread_id: "t1" });
    assert.match(req.headers.get("idempotency-key") ?? "", UUID);
  });

  test("omits run and thread ids when not given", async () => {
    const { tc, fake } = client([json(200, OK)]);
    await tc.invoke("add", {});
    assert.deepEqual(JSON.parse(fake.requests[0]!.body!), { arguments: {} });
  });

  test("uses the caller's Idempotency-Key", async () => {
    const { tc, fake } = client([json(200, OK)]);
    await tc.invoke("add", {}, { idempotencyKey: "order-42" });
    assert.equal(fake.requests[0]!.headers.get("idempotency-key"), "order-42");
  });

  test("keeps one Idempotency-Key across its retries and a new one per call", async () => {
    const { tc, fake } = client([networkError("ECONNREFUSED"), json(200, OK), json(200, OK)]);
    await tc.invoke("add", {});
    await tc.invoke("add", {});
    const keys = fake.requests.map((r) => r.headers.get("idempotency-key"));
    assert.equal(keys[0], keys[1]);
    assert.notEqual(keys[1], keys[2]);
  });

  test("returns the pending result of a 202", async () => {
    const { tc } = client([json(202, PENDING, { location: "/aa/approvals/apr_1", "retry-after": "2" })]);
    assert.deepEqual(await tc.invoke("send", { to: "a" }), PENDING);
  });

  test("rejects a body that does not match the status", async () => {
    for (const step of [json(200, PENDING), json(202, OK), json(200, { status: "weird" }), json(200, [1])]) {
      const { tc } = client([step]);
      await assert.rejects(tc.invoke("add", {}), ToolProtocolError);
    }
  });

  test("raises ToolArgsInvalidError with the errors list", async () => {
    const { tc } = client([
      problem(422, "tool-args-invalid", { tool: "add", errors: [{ loc: ["a"], type: "missing" }] }),
    ]);
    await assert.rejects(tc.invoke("add", {}), (err: unknown) => {
      assert.ok(err instanceof ToolArgsInvalidError);
      assert.deepEqual(err.errors, [{ loc: ["a"], type: "missing" }]);
      return true;
    });
  });

  test("rejects bad arguments before sending", async () => {
    const { tc, fake } = client([]);
    for (const args of [null, [1], "x", 1]) {
      await assert.rejects(tc.invoke("add", args as never), TypeError);
    }
    await assert.rejects(tc.invoke("", {}), TypeError);
    await assert.rejects(tc.invoke("add", {}, { idempotencyKey: "" }), TypeError);
    assert.equal(fake.requests.length, 0);
  });
});

describe("approvals", () => {
  test("getApproval reads one approval", async () => {
    const { tc, fake } = client([json(200, APPROVAL)]);
    assert.deepEqual(await tc.getApproval("apr_1"), APPROVAL);
    assert.equal(fake.requests[0]!.url.pathname, "/aa/approvals/apr_1");
  });

  test("listApprovals filters by run and status", async () => {
    const { tc, fake } = client([json(200, { approvals: [APPROVAL] })]);
    assert.deepEqual(await tc.listApprovals({ runId: "run_1", status: "pending" }), [APPROVAL]);
    const url = fake.requests[0]!.url;
    assert.equal(url.pathname, "/aa/approvals");
    assert.equal(url.searchParams.get("run_id"), "run_1");
    assert.equal(url.searchParams.get("status"), "pending");
  });

  test("listApprovals rejects a body without a list", async () => {
    const { tc } = client([json(200, {})]);
    await assert.rejects(tc.listApprovals(), ToolProtocolError);
  });

  test("decide posts the decision with an Idempotency-Key and returns the result", async () => {
    const { tc, fake } = client([json(200, OK)]);
    const result = await tc.decide("apr_1", { approved: true, editedArguments: { to: "b" }, reason: "ok" });
    assert.deepEqual(result, OK);
    const req = fake.requests[0]!;
    assert.equal(req.url.pathname, "/aa/approvals/apr_1/decide");
    assert.deepEqual(JSON.parse(req.body!), { approved: true, edited_arguments: { to: "b" }, reason: "ok" });
    assert.match(req.headers.get("idempotency-key") ?? "", UUID);
  });

  test("decide returns a denial", async () => {
    const denied = { status: "denied", run_id: "run_1", call_id: "call_1", approval_id: "apr_1", reason: "no" };
    const { tc, fake } = client([json(200, denied)]);
    assert.deepEqual(await tc.decide("apr_1", { approved: false }, { idempotencyKey: "k1" }), denied);
    assert.deepEqual(JSON.parse(fake.requests[0]!.body!), { approved: false });
    assert.equal(fake.requests[0]!.headers.get("idempotency-key"), "k1");
  });

  test("decide raises ApprovalAlreadyDecidedError", async () => {
    const { tc } = client([problem(409, "approval-already-decided", { approval_id: "apr_1" })]);
    await assert.rejects(tc.decide("apr_1", { approved: true }), ApprovalAlreadyDecidedError);
  });

  test("decide rejects a pending or unknown body", async () => {
    const { tc } = client([json(200, PENDING)]);
    await assert.rejects(tc.decide("apr_1", { approved: true }), ToolProtocolError);
  });

  test("decide requires a boolean decision", async () => {
    const { tc } = client([]);
    await assert.rejects(tc.decide("apr_1", { approved: "yes" } as never), TypeError);
    await assert.rejects(tc.decide("", { approved: true }), TypeError);
  });
});

describe("cost", () => {
  test("cost reads the run summary", async () => {
    const summary = { run_id: "run 1", total_usd: 0.5, by_stage: {}, by_kind: {} };
    const { tc, fake } = client([json(200, summary)]);
    assert.deepEqual(await tc.cost("run 1"), summary);
    assert.equal(fake.requests[0]!.url.pathname, "/aa/runs/run%201/cost");
  });

  test("cost raises NotFoundError without a ledger", async () => {
    const { tc } = client([problem(404, "not-found")]);
    await assert.rejects(tc.cost("run_1"), NotFoundError);
  });
});

describe("secrets", () => {
  test("the token never shows in inspection or JSON", () => {
    const tc = new ToolClient({ baseUrl: "https://srv", token: "s3cr3t-token" });
    assert.ok(!inspect(tc, { depth: 10, showHidden: true }).includes("s3cr3t-token"));
    assert.ok(!JSON.stringify(tc).includes("s3cr3t-token"));
  });
});
