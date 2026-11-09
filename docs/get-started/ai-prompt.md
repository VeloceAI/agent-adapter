# Prompt for any AI

Paste this into Claude, ChatGPT, Gemini, Cursor or any coding assistant that can see your
project. It reads the guide, looks at your stack and tells you what to install and connect,
before it touches a file. Use the copy button on the top right of the box.

```text
--8<-- "docs/assets/ai-prompt.txt"
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
