# Integrations: model APIs, protocols and importers

Export tools to model APIs and agent protocols (MCP, A2A, AG-UI); import tools from LangChain, OpenAPI and MCP. Each module imports its SDK lazily: install the extra first ([Integrations](../integrations/index.md)).

## `agent_adapter.integrations.openai`

::: agent_adapter.integrations.openai
    options:
      members: [INTEGRATION, OpenAIIntegration, chat_tool, chat_tools, responses_tool, responses_tools]

## `agent_adapter.integrations.gemini`

::: agent_adapter.integrations.gemini
    options:
      members: [GeminiIntegration, INTEGRATION, afc_callable, afc_callables, function_declaration, function_declarations, genai_tool, run_function_calls]

## `agent_adapter.integrations.anthropic`

::: agent_adapter.integrations.anthropic
    options:
      members: [AnthropicIntegration, INTEGRATION, anthropic_tool, anthropic_tools, tool_result_blocks]

## `agent_adapter.integrations.langchain`

::: agent_adapter.integrations.langchain
    options:
      members: [INTEGRATION, LangChainIntegration, import_langchain, structured_tool, structured_tools]

## `agent_adapter.integrations.openapi`

::: agent_adapter.integrations.openapi
    options:
      members: [Auth, BearerAuth, HeaderAuth, INTEGRATION, OpenAPIIntegration, QueryAuth, import_openapi]

## `agent_adapter.integrations.openapi.auth`

`agent_adapter.integrations.openapi.auth` re-exports, documented under the canonical path: [`Auth`](#agent_adapter.integrations.openapi.Auth), [`BearerAuth`](#agent_adapter.integrations.openapi.BearerAuth), [`HeaderAuth`](#agent_adapter.integrations.openapi.HeaderAuth), [`QueryAuth`](#agent_adapter.integrations.openapi.QueryAuth).

## `agent_adapter.integrations.mcp`

::: agent_adapter.integrations.mcp
    options:
      members: [APPROVAL_ID_META, CALL_ID_META, ERROR_KIND_META, INTEGRATION, MCPIntegration, MemoryTaskStore, MemoryTokenStorage, RUN_ID_META, RedisTaskStore, TASKS_EXTENSION, TASK_ID_META, TaskOptions, TaskStore, client_metadata_document, import_mcp, mcp_client, mcp_server, mcp_tool, oauth_client, output_schema_for, streamable_http_app, tool_annotations]

## `agent_adapter.integrations.mcp.annotations`

`agent_adapter.integrations.mcp.annotations` re-exports, documented under the canonical path: [`APPROVAL_ID_META`](#agent_adapter.integrations.mcp.APPROVAL_ID_META), [`CALL_ID_META`](#agent_adapter.integrations.mcp.CALL_ID_META), [`ERROR_KIND_META`](#agent_adapter.integrations.mcp.ERROR_KIND_META), [`RUN_ID_META`](#agent_adapter.integrations.mcp.RUN_ID_META), [`output_schema_for`](#agent_adapter.integrations.mcp.output_schema_for), [`tool_annotations`](#agent_adapter.integrations.mcp.tool_annotations).

## `agent_adapter.integrations.mcp.client`

`agent_adapter.integrations.mcp.client` re-exports, documented under the canonical path: [`import_mcp`](#agent_adapter.integrations.mcp.import_mcp), [`mcp_client`](#agent_adapter.integrations.mcp.mcp_client).

## `agent_adapter.integrations.mcp.oauth`

`agent_adapter.integrations.mcp.oauth` re-exports, documented under the canonical path: [`MemoryTokenStorage`](#agent_adapter.integrations.mcp.MemoryTokenStorage), [`client_metadata_document`](#agent_adapter.integrations.mcp.client_metadata_document), [`oauth_client`](#agent_adapter.integrations.mcp.oauth_client).

## `agent_adapter.integrations.mcp.server`

::: agent_adapter.integrations.mcp.server
    options:
      members: [AUTHENTICATED, TARGET, require_mcp]

Also exported here, documented under the canonical path: [`mcp_server`](#agent_adapter.integrations.mcp.mcp_server), [`mcp_tool`](#agent_adapter.integrations.mcp.mcp_tool), [`streamable_http_app`](#agent_adapter.integrations.mcp.streamable_http_app).

## `agent_adapter.integrations.mcp.tasks`

::: agent_adapter.integrations.mcp.tasks
    options:
      members: [APPROVAL_URL_PLACEHOLDER]

Also exported here, documented under the canonical path: [`MemoryTaskStore`](#agent_adapter.integrations.mcp.MemoryTaskStore), [`RedisTaskStore`](#agent_adapter.integrations.mcp.RedisTaskStore), [`TASKS_EXTENSION`](#agent_adapter.integrations.mcp.TASKS_EXTENSION), [`TASK_ID_META`](#agent_adapter.integrations.mcp.TASK_ID_META), [`TaskOptions`](#agent_adapter.integrations.mcp.TaskOptions), [`TaskStore`](#agent_adapter.integrations.mcp.TaskStore).

## `agent_adapter.integrations.a2a`

::: agent_adapter.integrations.a2a
    options:
      members: [A2AConnection, A2AIntegration, CARD_PATH, Caller, HostRefused, INTEGRATION, PRINCIPAL_STATE, ResponseTooLarge, TOOLS_EXTENSION, ToolAgent, a2a_app, a2a_executor, agent_card, agent_skill, bounded_task_store, call_remote, connect_a2a, import_a2a, static_bearer]

## `agent_adapter.integrations.ag_ui`

::: agent_adapter.integrations.ag_ui
    options:
      members: [AGUIBridge, AGUIInputInvalid, AGUIIntegration, AGUIRun, Agent, Authenticate, INTEGRATION, InterruptTable, PendingCall, TARGET, agui_endpoint, anonymous, parse_input, to_messages]
