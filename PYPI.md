# Agent Adapter

**Build with any framework. Ship with one safety layer.**

Write a tool once, as a plain Python function. Every call to it, from any model or agent
framework, is approved, budgeted, priced and traced the same way.

LangChain, CrewAI or ADK help you build an agent. Agent Adapter makes every tool call that
agent makes safe, priced and traced, in any framework or in none.

It turns your functions into LLM tools, checks their schemas once at startup, and runs
every call through one executor that applies approval, budgets, guards, caching and cost
tracking. The core depends only on pydantic. Every framework is an optional extra, imported
lazily. No agent framework is needed: `run_tool_loop` with your own model is the agent loop,
and `Flow` is the graph.

## Install

Python 3.11 or later.

```bash
pip install agent-adapter                 # core (pydantic only)
pip install "agent-adapter[openai]"       # + the extras you need
```

## Quickstart

```python
import agent_adapter as aa


@aa.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


print(aa.call(add, a=1, b=2))  # 3: validated and guarded by the default executor
```

`@aa.tool` registers the function; `aa.call` runs it through the default executor. From
async code (FastAPI, ...) use `await aa.acall(add, a=1, b=2)`.

A tool with side effects asks a person first, once, and the approval survives a restart:

```python
@aa.tool(confirm=True)
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order_id}"


executor = aa.configure(approval=aa.DeferredApproval.sqlite("approvals.db"))
try:
    aa.call(refund, order_id="o-7", amount=500)
except aa.ApprovalPending as pending:
    approval_id = pending.approval_id  # show it to a reviewer
# ... later, in any process that opens the same file:
print(executor.resume_sync(approval_id, True, decided_by="ana").value)
```

## Why Agent Adapter

Each framework has its own tool format, its own idea of approval and no shared view of cost.
Change framework, or run two, and the rules are rewritten. Agent Adapter keeps them in one place.

| You need | Without it | With Agent Adapter |
|---|---|---|
| A person to approve risky actions | each framework's own flag, easy to bypass | a one-use approval the executor checks on every call, from any source; it survives a restart |
| Spending limits | prices added up by hand | `BudgetHook` per run, user or team, per day or month; every call priced in a ledger |
| Rules you can prove | settings spread across code | one policy file for access, approvals, guards and budgets; a signed, hash-chained audit log |
| To change framework | rewrite the tools | keep them: one module per framework, the same tools everywhere |

## Works with

| You have | Use |
|---|---|
| a script, a service, a notebook | `await executor.run(...)`; `run_sync(...)` from sync code (Flask, Django, Celery) |
| a raw model API | OpenAI, Gemini and Anthropic renderers, or the built-in `run_tool_loop` |
| an agent framework | one extra each: LangGraph, LangChain, ADK, OpenAI Agents SDK, Pydantic AI, LlamaIndex, CrewAI, Agent Framework, Agno, Strands, Haystack, Claude Agent SDK, Semantic Kernel, DSPy, smolagents, AutoGen, CAMEL-AI, AG2 |
| a durable engine | Temporal or DBOS run a `Flow` crash-safe |
| another agent, UI or language | MCP, an HTTP tool server, A2A, AG-UI and a TypeScript client |

## Features

| Area | What you get |
|---|---|
| Tools | `@registry.tool`, `registry.wrap()`, shaping, a portable-schema check, typed errors |
| Execution | `run`, `run_many`, `run_plan`, `parallel`; retry, timeout, hooks, run guards |
| Approval | one-use decisions; stores in SQLite, Redis, SQLAlchemy and DynamoDB; batch approval |
| Guardrails | `PIIGuard` (emails, phones, cards, IBANs, secrets), Presidio, NeMo, your own guard |
| Budgets and cost | per run, user or team budgets; a priced ledger; tool, response and prefix caches |
| Policy and audit | one policy file per tool, team and environment; a signed, hash-chained audit log |
| Flows and agents | route, map, join, pause, sub-flows, node cache; agents as tools and handoffs |
| Memory and RAG | per-user memory (SQLite, Mem0, Zep, Graphiti); a retriever tool with citations |
| Sandboxed code | `code_tool` over Docker or a hosted sandbox; every run needs approval |
| Evals | `run_eval` with graders, an LLM judge, reports, a pytest plugin and a cost advisor |
| Observability | an event catalogue, OpenTelemetry GenAI spans, and a local run viewer with a chat page |

## Safe by default

- A `confirm` tool runs only after an approval port approves, and each decision is used
  once. No port: denied. A framework's own "approved" signal never counts alone.
- The HTTP server refuses to start without an authenticator unless anonymous mode is chosen.
- The local viewer and chat bind 127.0.0.1 only, behind a private token link.
- Logs never carry secrets or the arguments of `confirm` tools.

## Command line

| Command | Does |
|---|---|
| `agent-adapter init fastapi my-service` | writes a starter project from a tested template |
| `agent-adapter doctor` | checks Python, extras, integrations and the pip index |
| `agent-adapter ui` | the local run viewer; with `--agent` or `--model`, a chat page |
| `agent-adapter eval app:agent cases.jsonl` | grades an agent on eval cases |
| `agent-adapter guide` | prints the guide for AI coding agents |

## Licence

Apache License 2.0. Copyright 2026 Veloce AI. Documentation:
<https://agent-adapter.veloceai.in/>
