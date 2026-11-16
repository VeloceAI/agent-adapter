// Each framework's own MCP client against our public app: list the tools, call one, and
// see a confirm tool park instead of running. `tests/docs/test_connect_js.py` runs it.
import { createMCPClient } from "@ai-sdk/mcp";
import { createMcpHost } from "@genkit-ai/mcp";
import { MCPClient } from "@mastra/mcp";
import { MCPServerStreamableHttp } from "@openai/agents";
import { genkit } from "genkit";

const url = `${process.env.AA_URL}/mcp`;
const headers = { Authorization: `Bearer ${process.env.AA_AGENT_TOKEN}` };
const rate = { region: "us" };
const invoice = { customer_id: "cus_1024", amount: 49.5 };

function text(result: unknown): unknown {
  const { content } = (result ?? {}) as { content?: { type: string; text?: string }[] };
  return content ? content.map((part) => part.text ?? "").join("") : result;
}

function report(name: string, tools: string[], value: unknown, parked: unknown): void {
  const pending = JSON.stringify(parked).includes("approval required: apr_");
  console.log(`${name}: ${tools.sort().join(",")} | ${JSON.stringify(value)} | parked=${pending}`);
}

async function vercel(): Promise<void> {
  const mcp = await createMCPClient({ transport: { type: "http", url, headers } });
  try {
    const tools = await mcp.tools();
    const options = { toolCallId: "c1", messages: [] };
    const value = await tools.shipping_rate.execute!(rate, options);
    const parked = await tools.send_invoice.execute!(invoice, { ...options, toolCallId: "c2" });
    report("vercel", Object.keys(tools), text(value), parked);
  } finally {
    await mcp.close();
  }
}

async function mastra(): Promise<void> {
  const mcp = new MCPClient({
    servers: { shopTools: { url: new URL(url), requestInit: { headers } } },
    timeout: 60_000,
  });
  try {
    const tools = await mcp.listTools();
    const call = (name: string, args: object) => tools[`shopTools_${name}`].execute!(args, {});
    const value = await call("shipping_rate", rate);
    const parked = await call("send_invoice", invoice).catch((err: Error) => err.message); // throws
    report("mastra", Object.keys(tools), text(value), parked);
  } finally {
    await mcp.disconnect();
  }
}

async function openaiAgents(): Promise<void> {
  const server = new MCPServerStreamableHttp({ url, name: "shop_tools", requestInit: { headers } });
  await server.connect();
  try {
    const tools = (await server.listTools()).map((tool) => tool.name);
    const value = await server.callTool("shipping_rate", rate);
    report("openai-agents", tools, text({ content: value }), await server.callTool("send_invoice", invoice));
  } finally {
    await server.close();
  }
}

async function genkitHost(): Promise<void> {
  const ai = genkit({});
  const host = createMcpHost({
    name: "agentAdapter",
    mcpServers: { shopTools: { url, requestInit: { headers } } },
  });
  try {
    const tools = await host.getActiveTools(ai);
    const byName = Object.fromEntries(tools.map((tool) => [tool.__action.name, tool]));
    const value = await byName["shopTools/shipping_rate"](rate);
    const parked = await byName["shopTools/send_invoice"](invoice).catch((err: Error) => err.message);
    report("genkit", Object.keys(byName), value, parked);
  } finally {
    await host.close();
  }
}

await vercel();
await mastra();
await openaiAgents();
await genkitHost();
