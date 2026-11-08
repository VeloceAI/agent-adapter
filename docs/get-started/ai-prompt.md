# Prompt for any AI

Paste this into Claude, ChatGPT, Gemini, Cursor or any coding assistant that can see your
project. It reads the guide, looks at your stack and tells you what to install and connect,
before it touches a file. Use the copy button on the top right of the box.

```text
Help me add Agent Adapter (Python package "agent-adapter") to my project.

1. Read the guide first: https://agent-adapter.veloceai.in/llms-full.txt
   If you cannot open links, ask me to paste the output of `agent-adapter guide`.

2. Look at my project and list what you find:
   - Python version and package manager
   - agent framework, if any (LangGraph, LangChain, CrewAI, OpenAI Agents SDK,
     Pydantic AI, LlamaIndex, MCP, ...), or none
   - backend or runtime (FastAPI, Django, Flask, Celery, AWS Lambda, scripts, notebooks)
   - which LLM providers it calls
   - functions that should become tools, and which of them write, pay, send or delete

3. Then give me a short plan:
   - the exact pip install line, with the extras my stack needs
   - which functions get @tool, and which need confirm=True (a person approves first)
   - how to connect them: aa.attach("<framework>", registry, executor=executor),
     or the executor on its own if I use no framework
   - how to call tools in my backend: await executor.run(...) in async code,
     executor.run_sync(...) in sync code (Celery, scripts, sync views)
   - the limits, cost tracking and traces to turn on (RunLimits, BudgetHook)

4. Use only names that appear in the guide; do not invent APIs.
   Show me the plan and wait for my yes before you change any file.
```

## What you get back

| Your stack | The AI should suggest |
|---|---|
| A framework (LangChain, CrewAI, ...) | its extra, `@tool` on your functions, one `aa.attach(...)` line |
| No framework | the executor alone as your agent loop, with a provider extra such as `openai` |
| FastAPI or async Django | `await executor.run(...)` in the route or view |
| Celery, Flask, scripts | `executor.run_sync(...)` |
| Tools that change things | `confirm=True` plus an approval port on the executor |

## When the AI works in your repo every day

Give it the guide as a file instead of a link, so it is always there and matches your
installed version: see [For AI coding agents](for-ai-agents.md).
