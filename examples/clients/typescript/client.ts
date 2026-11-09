// Call the tool server from TypeScript: MCP with the official SDK, then plain REST.
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const url: string = process.env.AA_URL ?? "http://localhost:8000";
const headers = { Authorization: `Bearer ${process.env.AA_TOKEN ?? "dev-token"}` };

const client = new Client({ name: "agent-adapter-client", version: "1.0.0" });
await client.connect(new StreamableHTTPClientTransport(new URL(`${url}/mcp`), { requestInit: { headers } }));
const { tools } = await client.listTools();
if (!tools.some((tool) => tool.name === "get_weather")) throw new Error("get_weather missing");
const result = await client.callTool({ name: "get_weather", arguments: { city: "Paris" } });
console.log("MCP:", (result.content as { text: string }[])[0].text);
await client.close();

const res = await fetch(`${url}/tools/get_weather/invoke`, {
  method: "POST",
  headers: { ...headers, "Content-Type": "application/json" },
  body: JSON.stringify({ arguments: { city: "Paris" } }),
});
console.log("REST:", (await res.json()).result);
