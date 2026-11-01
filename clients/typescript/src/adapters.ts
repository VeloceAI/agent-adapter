/** Tool shapes for TypeScript agent frameworks, built from the server's catalogue. */
// The functional form of `ToolClient.toolDefinitions` / `asVercelTools` (§16); takes any `ToolInvoker`.
import type { InvokeOptions, ToolDefinition, ToolDefinitionsOptions, ToolInvoker } from "./types.ts";

/** One neutral definition per catalogue tool; `execute` returns `result`, or `{ pending_approval }`. */
export async function toolDefinitions(
  client: ToolInvoker,
  options: ToolDefinitionsOptions = {},
): Promise<ToolDefinition[]> {
  const tools = await client.listTools(options.tags ? { tags: options.tags } : {});
  return tools.map((descriptor) => ({
    name: descriptor.name,
    description: descriptor.description,
    inputSchema: descriptor.input_schema,
    execute: async (input, executeOptions) => {
      const invokeOptions: InvokeOptions = {};
      if (options.runId !== undefined) invokeOptions.runId = options.runId;
      if (options.threadId !== undefined) invokeOptions.threadId = options.threadId;
      if (executeOptions?.abortSignal !== undefined) invokeOptions.signal = executeOptions.abortSignal;
      const result = await client.invoke(descriptor.name, input, invokeOptions);
      return result.status === "ok" ? result.result : { pending_approval: result.approval_id };
    },
  }));
}

/** Vercel AI SDK tools keyed by name: `asVercelTools(client, tool, jsonSchema)` with the SDK's helpers. */
export async function asVercelTools(
  client: ToolInvoker,
  tool: (def: any) => any,
  jsonSchema: (schema: any) => any,
  options: ToolDefinitionsOptions = {},
): Promise<Record<string, unknown>> {
  const definitions = await toolDefinitions(client, options);
  // fromEntries defines own properties, so a tool named `__proto__` cannot touch the prototype
  return Object.fromEntries(
    definitions.map((def) => [
      def.name,
      tool({ description: def.description, inputSchema: jsonSchema(def.inputSchema), execute: def.execute }),
    ]),
  );
}
