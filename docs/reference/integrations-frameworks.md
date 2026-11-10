# Integrations: agent frameworks

Export tools, approval and flows to agent frameworks. Each module imports its SDK lazily: install the extra first ([Integrations](../integrations/index.md)).

## `agent_adapter.integrations.langgraph`

::: agent_adapter.integrations.langgraph
    options:
      members: [INTEGRATION, LangGraphApproval, LangGraphCompressor, LangGraphIntegration, ToolNode, ToolTask, compile_flow, fan_out, langgraph_compressor, one_use, reducer, run_compiled, tool_node, tool_task]

## `agent_adapter.integrations.adk`

::: agent_adapter.integrations.adk
    options:
      members: [AdkIntegration, INTEGRATION, TARGET, adk_tool, adk_tools, bind_context, compile_flow, resolve_confirmations, resume_message, run_workflow]

## `agent_adapter.integrations.openai_agents`

::: agent_adapter.integrations.openai_agents
    options:
      members: [INTEGRATION, OpenAIAgentsIntegration, function_tool, function_tools, resolve_interruptions]

## `agent_adapter.integrations.pydantic_ai`

::: agent_adapter.integrations.pydantic_ai
    options:
      members: [INTEGRATION, PydanticAIIntegration, pydantic_ai_history_processor, pydantic_ai_tool, pydantic_ai_tools, resolve_deferred]

## `agent_adapter.integrations.llamaindex`

::: agent_adapter.integrations.llamaindex
    options:
      members: [INTEGRATION, LlamaIndexIntegration, llamaindex_tool, llamaindex_tools]

## `agent_adapter.integrations.crewai`

::: agent_adapter.integrations.crewai
    options:
      members: [CrewAIIntegration, INTEGRATION, crewai_tool, crewai_tools]

## `agent_adapter.integrations.agent_framework`

::: agent_adapter.integrations.agent_framework
    options:
      members: [AgentFrameworkIntegration, INTEGRATION, agent_framework_tool, agent_framework_tools, resolve_approvals]

## `agent_adapter.integrations.agno`

::: agent_adapter.integrations.agno
    options:
      members: [AgnoIntegration, INTEGRATION, agno_compressed_model, agno_function, agno_functions, resolve_requirements]

## `agent_adapter.integrations.strands`

::: agent_adapter.integrations.strands
    options:
      members: [INTEGRATION, StrandsIntegration, strands_approval_hook, strands_compress_middleware, strands_tool, strands_tools]

## `agent_adapter.integrations.haystack`

::: agent_adapter.integrations.haystack
    options:
      members: [HaystackIntegration, INTEGRATION, confirmation_hook, haystack_tool, haystack_tools]

## `agent_adapter.integrations.claude_agent`

::: agent_adapter.integrations.claude_agent
    options:
      members: [ClaudeAgentIntegration, ClaudeSdkBridge, INTEGRATION, can_use_tool, claude_sdk_bridge, claude_sdk_server, claude_sdk_tool, claude_sdk_tools]

## `agent_adapter.integrations.semantic_kernel`

::: agent_adapter.integrations.semantic_kernel
    options:
      members: [INTEGRATION, SemanticKernelIntegration, approval_filter, kernel_function, kernel_plugin]

## `agent_adapter.integrations.dspy`

::: agent_adapter.integrations.dspy
    options:
      members: [DSPyIntegration, INTEGRATION, dspy_compress_prompt, dspy_tool, dspy_tools]

## `agent_adapter.integrations.smolagents`

::: agent_adapter.integrations.smolagents
    options:
      members: [INTEGRATION, SmolagentsIntegration, smolagents_compressed_model, smolagents_tool, smolagents_tools]

## `agent_adapter.integrations.autogen`

::: agent_adapter.integrations.autogen
    options:
      members: [AutoGenCall, AutoGenIntegration, INTEGRATION, autogen_model_context, autogen_tool, autogen_tools]

## `agent_adapter.integrations.camel`

::: agent_adapter.integrations.camel
    options:
      members: [CamelIntegration, INTEGRATION, camel_tool, camel_tools, openai_tool_schema]

## `agent_adapter.integrations.ag2`

::: agent_adapter.integrations.ag2
    options:
      members: [AG2Integration, INTEGRATION, ag2_compress_policy, ag2_tool, ag2_tools]

## `agent_adapter.integrations.adk.compile`

`agent_adapter.integrations.adk.compile` re-exports, documented under the canonical path: [`compile_flow`](#agent_adapter.integrations.adk.compile_flow).

## `agent_adapter.integrations.adk.tools`

`agent_adapter.integrations.adk.tools` re-exports, documented under the canonical path: [`adk_tool`](#agent_adapter.integrations.adk.adk_tool), [`adk_tools`](#agent_adapter.integrations.adk.adk_tools), [`bind_context`](#agent_adapter.integrations.adk.bind_context), [`resolve_confirmations`](#agent_adapter.integrations.adk.resolve_confirmations).

## `agent_adapter.integrations.langgraph.approval`

`agent_adapter.integrations.langgraph.approval` re-exports, documented under the canonical path: [`LangGraphApproval`](#agent_adapter.integrations.langgraph.LangGraphApproval).

## `agent_adapter.integrations.langgraph.compile`

`agent_adapter.integrations.langgraph.compile` re-exports, documented under the canonical path: [`compile_flow`](#agent_adapter.integrations.langgraph.compile_flow), [`run_compiled`](#agent_adapter.integrations.langgraph.run_compiled).

## `agent_adapter.integrations.langgraph.tools`

`agent_adapter.integrations.langgraph.tools` re-exports, documented under the canonical path: [`ToolNode`](#agent_adapter.integrations.langgraph.ToolNode), [`ToolTask`](#agent_adapter.integrations.langgraph.ToolTask), [`fan_out`](#agent_adapter.integrations.langgraph.fan_out), [`tool_node`](#agent_adapter.integrations.langgraph.tool_node), [`tool_task`](#agent_adapter.integrations.langgraph.tool_task).
