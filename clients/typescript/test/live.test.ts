/**
 * Live suite against a running tool server (the MC gate, `nox -s ts`). Skipped unless AA_BASE_URL is set.
 *
 * Env: AA_BASE_URL; AA_TOKEN (bearer, optional); AA_LIVE_TOOL / AA_LIVE_ARGS (a tool that runs
 * without approval, default: the first such tool with no required arguments); AA_LIVE_CONFIRM_TOOL /
 * AA_LIVE_CONFIRM_ARGS (a tool with approval "always"); AA_LIVE_STRICT=1 fails instead of skipping
 * when the server's registry has no tool for a scenario.
 */
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { describe, test } from "node:test";
import type { TestContext } from "node:test";

import {
  ApprovalAlreadyDecidedError,
  NotFoundError,
  ToolArgsInvalidError,
  ToolClient,
  ToolNotFoundError,
  UnauthorizedError,
} from "../src/index.ts";
import type { AdapterEvent, ClientOptions, LedgerSummary, RunEvent, ToolDescriptor } from "../src/index.ts";

const BASE = process.env["AA_BASE_URL"];
const TOKEN = process.env["AA_TOKEN"];
const STRICT = process.env["AA_LIVE_STRICT"] === "1";

function options(): ClientOptions {
  return { baseUrl: BASE ?? "http://unused", timeoutMs: 15_000, ...(TOKEN ? { token: TOKEN } : {}) };
}

function required(tool: ToolDescriptor): string[] {
  const value = tool.input_schema["required"];
  return Array.isArray(value) ? (value as string[]) : [];
}

function argsFrom(name: string): Record<string, unknown> {
  const raw = process.env[name];
  return raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
}

function pick(
  t: TestContext,
  tools: ToolDescriptor[],
  envName: string,
  fallback: (tool: ToolDescriptor) => boolean,
): ToolDescriptor | undefined {
  const wanted = process.env[envName];
  const tool = wanted ? tools.find((x) => x.name === wanted) : tools.find(fallback);
  if (tool === undefined) {
    assert.ok(!STRICT, `no tool for this scenario (set ${envName})`);
    t.skip(`registry has no tool for this scenario (set ${envName})`);
  }
  return tool;
}

const runId = (): string => `run_ts${randomUUID().replaceAll("-", "")}`;

describe("live tool server", { skip: BASE ? false : "AA_BASE_URL not set" }, () => {
  const tc = new ToolClient(options());

  test("health and catalogue", async () => {
    assert.equal((await tc.health()).status, "ok");
    const tools = await tc.listTools();
    assert.ok(tools.length > 0, "the registry is empty");
    const first = tools[0]!;
    assert.equal((await tc.getTool(first.name)).name, first.name);
    const shaped = await tc.listTools({ shape: "openai" });
    assert.ok(shaped.every((tool) => tool.provider !== undefined));
  });

  test("unknown tool", async () => {
    await assert.rejects(tc.getTool("no_such_tool_ts_live"), ToolNotFoundError);
  });

  test("missing credentials", { skip: TOKEN ? false : "AA_TOKEN not set" }, async () => {
    const anonymous = new ToolClient({ baseUrl: BASE ?? "", retry: false });
    await assert.rejects(anonymous.listTools(), UnauthorizedError);
  });

  test("invoke, idempotent replay and cost", async (t) => {
    const tool = pick(
      t,
      await tc.listTools(),
      "AA_LIVE_TOOL",
      (x) => x.approval === "never" && required(x).length === 0,
    );
    if (!tool) return;
    const run = runId();
    const key = randomUUID();
    const first = await tc.invoke(tool.name, argsFrom("AA_LIVE_ARGS"), { runId: run, idempotencyKey: key });
    assert.equal(first.status, "ok");
    const again = await tc.invoke(tool.name, argsFrom("AA_LIVE_ARGS"), { runId: run, idempotencyKey: key });
    assert.equal(again.call_id, first.call_id);
    try {
      const summary: LedgerSummary = await tc.cost(run);
      assert.equal(typeof summary.total_usd, "number");
    } catch (err) {
      assert.ok(err instanceof NotFoundError, "cost fails only when the server has no ledger");
    }
  });

  test("invalid arguments", async (t) => {
    const tool = pick(t, await tc.listTools(), "AA_LIVE_STRICT_TOOL", (x) => required(x).length > 0);
    if (!tool) return;
    await assert.rejects(tc.invoke(tool.name, {}, { runId: runId() }), ToolArgsInvalidError);
  });

  test("approval round trip with events", async (t) => {
    const tool = pick(
      t,
      await tc.listTools(),
      "AA_LIVE_CONFIRM_TOOL",
      (x) => x.approval === "always" && required(x).length === 0,
    );
    if (!tool) return;
    const run = runId();
    const pending = await tc.invoke(tool.name, argsFrom("AA_LIVE_CONFIRM_ARGS"), { runId: run });
    assert.equal(pending.status, "pending_approval");
    if (pending.status !== "pending_approval") return;
    const listed = await tc.listApprovals({ runId: run, status: "pending" });
    assert.ok(listed.some((a) => a.approval_id === pending.approval_id));
    assert.equal((await tc.getApproval(pending.approval_id)).status, "pending");

    const signal = AbortSignal.timeout(20_000);
    const events: RunEvent[] = [];
    const reading = (async () => {
      for await (const event of tc.events(run, { closeOn: ["tool.finished", "tool.failed"], signal })) {
        events.push(event);
      }
    })();
    const decided = await tc.decide(pending.approval_id, { approved: true, reason: "ts live suite" });
    assert.equal(decided.status, "ok");
    await reading;
    const types = events.map((e) => e.event);
    assert.ok(types.includes("tool.finished"), `events: ${types.join(", ")}`);
    for (const event of events) assert.equal((event.data as AdapterEvent).type, event.event);
    await assert.rejects(tc.decide(pending.approval_id, { approved: false }), ApprovalAlreadyDecidedError);
  });
});
