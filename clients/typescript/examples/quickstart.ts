/** Doc snippets for docs/cross-language/typescript-client.md; run by test/examples.test.ts. */
import { ToolArgsInvalidError, ToolClient } from "../src/index.ts";
import type { LedgerSummary, ToolDescriptor } from "../src/index.ts";

/** Connects and lists the math tools. */
export async function connect(
  baseUrl: string,
  token: string,
): Promise<{ client: ToolClient; tools: ToolDescriptor[] }> {
  // --8<-- [start:connect]
  const client = new ToolClient({
    baseUrl, // e.g. "https://tools.internal/api" (include the server prefix)
    token, // or an (async) function, called before every attempt
    timeoutMs: 10_000,
  });
  const tools = await client.listTools({ tags: ["math"] });
  // --8<-- [end:connect]
  return { client, tools };
}

/** Calls a tool and turns a schema failure into a value. */
export async function add(client: ToolClient, runId: string): Promise<number | string | unknown[]> {
  // --8<-- [start:invoke]
  try {
    const call = await client.invoke<{ sum: number }>("add", { a: 2, b: 3 }, { runId });
    if (call.status === "pending_approval") return call.approval_id; // see "Approvals"
    return call.result.sum;
  } catch (err) {
    if (err instanceof ToolArgsInvalidError) return err.errors; // 422: what was wrong, no input values
    throw err;
  }
  // --8<-- [end:invoke]
}

/** Runs a confirm tool through a reviewer's decision while watching the run's events. */
export async function approveAndWatch(
  client: ToolClient,
  runId: string,
): Promise<{ result: unknown; events: string[] }> {
  // --8<-- [start:approve]
  const call = await client.invoke("send_email", { to: "ops@example.com" }, { runId });
  if (call.status === "ok") return { result: call.result, events: [] };

  const watching = (async () => {
    const seen: string[] = [];
    const closeOn = ["tool.finished", "tool.failed"];
    for await (const event of client.events(runId, { closeOn, signal: AbortSignal.timeout(60_000) })) {
      seen.push(event.event); // event.data is the server's Event.to_dict()
    }
    return seen;
  })();

  const request = await client.getApproval(call.approval_id); // show request.arguments to the reviewer
  const decided = await client.decide(call.approval_id, {
    approved: true,
    editedArguments: { ...request.arguments, cc: "audit@example.com" },
    reason: "checked by ops",
  });
  // --8<-- [end:approve]
  const events = await watching;
  return { result: decided.status === "ok" ? decided.result : decided.reason, events };
}

/** Reads what the run cost. */
export async function runCost(client: ToolClient, runId: string): Promise<number> {
  // --8<-- [start:cost]
  const summary: LedgerSummary = await client.cost(runId);
  const usd = summary.total_usd; // by_stage, by_kind, cache hit rates, parallelism
  // --8<-- [end:cost]
  return usd;
}

/** Builds Vercel AI SDK tools; `ai` is `import * as ai from "ai"`. */
export async function vercelTools(
  client: ToolClient,
  ai: { tool: (def: any) => any; jsonSchema: (schema: any) => any },
  runId: string,
): Promise<Record<string, unknown>> {
  // --8<-- [start:vercel]
  const tools = await client.asVercelTools(ai.tool, ai.jsonSchema, { runId });
  // generateText({ model, tools, prompt }): a confirm tool returns { pending_approval: "apr_..." }
  // --8<-- [end:vercel]
  return tools;
}
