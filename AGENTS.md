# Agent Adapter: guide for AI coding agents

Version 1.1.0. Generated from the package's own code, so it matches
this install. Print it with `agent-adapter guide`. This is the concise guide; `agent-adapter guide --full` adds every signature.

## What it is

A Python SDK: write a tool once as a plain function, then run it from any model
API, agent framework, MCP, A2A or HTTP. One executor adds validation, approvals,
budgets, cache, retries and tracing. It works with no agent framework; it can be
your agent loop.

## Install

`agent-adapter` is on PyPI (Apache-2.0). Install the core, or the extras you need:

```bash
pip install "agent-adapter[openai]"
```

From a checkout, use `pip install -e ".[openai]"` instead. Check the setup with
`agent-adapter doctor`.

Python: `>=3.11`; tested on 3.11, 3.12, 3.13. Framework versions:
the newest release inside each pinned range is tested. An extra marked
"own environment" conflicts with another extra: give it its own venv.

| I need | pip command | Supported versions |
|---|---|---|
| tools, executor, approval, budgets, cache, flows, compression, plain backends, queues, serverless | `pip install agent-adapter` | `pydantic>=2.12,<3` |
| OpenAI models (`OpenAIModel`, also Ollama, vLLM, LM Studio) | `pip install "agent-adapter[openai]"` | `openai>=3.41,<4` |
| Gemini models (`GeminiModel`, `genai_tool()`) | `pip install "agent-adapter[gemini]"` | `google-genai>=2.30,<3` |
| Anthropic models (`AnthropicModel`) | `pip install "agent-adapter[anthropic]"` | `anthropic>=2.3,<3` |
| Redis cache and approval store | `pip install "agent-adapter[redis]"` | `redis>=8.1,<9` |
| OpenTelemetry spans (`OTelSink`) | `pip install "agent-adapter[otel]"` | `opentelemetry-api>=1.39,<2` |
| LangChain tools, export and import | `pip install "agent-adapter[langchain]"` | `langchain-core>=1.4,<2` |
| LangGraph tool nodes, approvals, compiled flows | `pip install "agent-adapter[langgraph]"` | `langgraph>=1.2.15,<2`, `langgraph-checkpoint>=4.3,<5` |
| Google ADK tools and flows | `pip install "agent-adapter[adk]"` | `google-adk>=2.13,<3`; own environment (opentelemetry<=1.44) |
| MCP server, client import, Streamable HTTP | `pip install "agent-adapter[mcp]"` | `mcp>=2.3,<3` |
| OpenAI Agents SDK tools | `pip install "agent-adapter[openai-agents]"` | `openai-agents>=0.24,<1` |
| Pydantic AI tools | `pip install "agent-adapter[pydantic-ai]"` | `pydantic-ai-slim>=2.60,<3` |
| LlamaIndex tools | `pip install "agent-adapter[llamaindex]"` | `llama-index-core>=0.14,<0.15` |
| CrewAI tools | `pip install "agent-adapter[crewai]"` | `crewai>=1.15,<2`; own environment (mcp<2) |
| tools from an OpenAPI document (`import_openapi`) | `pip install "agent-adapter[openapi]"` | `httpx>=0.27,<1`, `pyyaml>=6,<7` |
| Microsoft Agent Framework tools | `pip install "agent-adapter[agent-framework]"` | `agent-framework-core>=1.23,<2` |
| Agno tools | `pip install "agent-adapter[agno]"` | `agno>=3.1,<4` |
| DSPy tools | `pip install "agent-adapter[dspy]"` | `dspy>=3.4,<4`; own environment (LiteLLM) |
| Strands Agents tools | `pip install "agent-adapter[strands]"` | `strands-agents>=1.60,<2`; own environment (mcp<2.2) |
| Haystack tools | `pip install "agent-adapter[haystack]"` | `haystack-ai>=3.3,<4` |
| smolagents tools | `pip install "agent-adapter[smolagents]"` | `smolagents>=1.24,<2` |
| Claude Agent SDK tools | `pip install "agent-adapter[claude-agent]"` | `claude-agent-sdk>=0.2.150,<0.3`, `mcp>=2.3,<3` |
| AutoGen tools | `pip install "agent-adapter[autogen]"` | `autogen-core>=0.7.5,<0.8`, `autogen-agentchat>=0.7.5,<0.8` |
| Semantic Kernel functions | `pip install "agent-adapter[semantic-kernel]"` | `semantic-kernel>=1.44,<2`; own environment (pydantic<2.14) |
| CAMEL-AI tools | `pip install "agent-adapter[camel]"` | `camel-ai>=0.2.90,<0.3`, `mcp<2`; own environment (needs mcp<2; pins pydantic<=2.12.0, while google-genai in `all` needs >=2.12.5) |
| AG2 tools | `pip install "agent-adapter[ag2]"` | `ag2>=1.2.2,<2` |
| A2A server and client | `pip install "agent-adapter[a2a]"` | `a2a-sdk[http-server]>=1.3,<2` |
| AG-UI event stream (CopilotKit and other UIs) | `pip install "agent-adapter[ag-ui]"` | `ag-ui-protocol>=1.0,<2` |
| Postgres or any SQLAlchemy database as the approval store | `pip install "agent-adapter[sqlalchemy]"` | `sqlalchemy[asyncio]>=2.0.30,<3` |
| DynamoDB approval store (AWS serverless) | `pip install "agent-adapter[dynamodb]"` | `aioboto3>=13,<16` |
| flows as Temporal workflows (`agent_adapter.integrations.temporal`) | `pip install "agent-adapter[temporal]"` | `temporalio>=1.35,<2` |
| flows as DBOS workflows (`DbosFlows`) | `pip install "agent-adapter[dbos]"` | `dbos>=3.4,<4` |
| HTTP tool server (ASGI) with MCP mounted | `pip install "agent-adapter[server]"` | `starlette>=1.0,<2`, `uvicorn>=0.30,<1`, `h11>=0.16,<1`, extra `mcp` |
| HTTP tool server under WSGI (Flask, Django) | `pip install "agent-adapter[server-wsgi]"` | `a2wsgi>=1.10,<2`, extra `server` |
| Presidio PII guard (`PresidioGuard`; also needs a spaCy model) | `pip install "agent-adapter[presidio]"` | `presidio-analyzer>=2.2,<3`, `presidio-anonymizer>=2.2,<3` |
| NeMo Guardrails guard (`NemoGuard`) | `pip install "agent-adapter[nemo]"` | `nemoguardrails>=0.24,<1` |
| Mem0 memory backend (`Mem0Memory`) | `pip install "agent-adapter[mem0]"` | `mem0ai>=2.2,<3` |
| Zep memory backend (`ZepMemory`) | `pip install "agent-adapter[zep]"` | `zep-cloud>=4.1,<5` |
| Graphiti knowledge-graph memory (`GraphitiMemory`) | `pip install "agent-adapter[graphiti]"` | `graphiti-core>=0.31,<1` |
| every extra that installs together in one environment | `pip install "agent-adapter[all]"` | every extra above except `adk`, `camel`, `crewai`, `dbos`, `dspy`, `graphiti`, `mem0`, `nemo`, `presidio`, `semantic-kernel`, `strands`, `temporal`, `zep` |

## Import rules

Import rules: everyday names come from `agent_adapter`; every other error from `agent_adapter.errors`; everything else from its subpackage (`agent_adapter.exec`, `.hooks`, `.approval`, ...); framework exports from `agent_adapter.integrations.<framework>`. Never import from `agent_adapter.core` or a `_private` module.

## Safety rules for agents writing code with it

Do:

- Mark every tool with a side effect (write, pay, send, delete) `confirm=True`,
  imported ones too: `import_openapi(..., operations=[...], confirm=True)` for the
  writes, and the same for `import_mcp` and `import_a2a` tools that change data.
- Give the executor an approval port: `CallbackApproval`, `DeferredApproval` or
  `AutoDeny`. With none, a `confirm=True` call is denied (it fails closed).
- Approve only through the executor: its approval port, or
  `executor.resume(approval_id, decision)`. A framework's own "approved" never
  counts alone; only a one-use decision issued by the adapter does.
- Run tools through the executor (`executor.run`, `run_sync`, `aa.call`, or
  `attach(..., executor=executor)`). Calling the function directly skips validation,
  approval, limits and the ledger.
- Pass `executor=executor` to every server and app (`create_app`, MCP, A2A,
  `attach`); without it they use the default executor, not your approval port.
- Set limits: `RunLimits(max_calls=)`, `BudgetHook(max_usd)`, and `timeout=` on
  slow tools.
- Raise `ToolError("...")` for a message the model should see; other exceptions
  reach the model only as `tool-failed`.
- Require auth on every server: `bearer_auth` (HTTP), `static_bearer` (A2A),
  `auth=` with `token_verifier=` (MCP). Without auth an app serves loopback only.
- Keep `run_id` and the call id with an approval id, so a resume matches its call.

Don't:

- Never pass an approval to `attach()`; approval belongs to the executor
  (`attach(..., approval=...)` raises `ToolUsageError`).
- Never log, print or return secrets, tokens, passwords or full personal data.
  `LoggingHook` redacts secret-named keys; `agent_adapter.hooks.redact` does it
  by hand.
- Never make a secret a model-visible argument: bind it with `fixed=` or read it
  from `ToolContext`.
- Never use `cache_scope="global"` for per-user, secret or personal data: global
  means every caller of the cache gets the same result.
- Never set `retries=` on a tool with a side effect unless it is idempotent: a
  timed-out sync tool keeps running in its thread, so a retry can repeat it.
- Never import from `agent_adapter.core` or a `_private` module.
- Never use `AutoApprove` for side-effect tools outside tests.
- Never add `--extra-index-url` for this package (dependency confusion).
- Never serve with `allow_anonymous` on a public host.

## I want to… → use…

Every snippet runs after this set-up (a test runs each one, or compiles it and checks
it against the API when its framework is not installed). Lines with `await` go in an
`async def`; in sync code, wrap them with `aa.run_sync()`.

```python
import agent_adapter as aa

registry = aa.Registry()


@registry.tool
def get_weather(city: str) -> str:
    """Today's weather for a city."""
    return f"{city}: sunny, 21 C"


@registry.tool(confirm=True)  # a side effect: runs only after an approval
def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    return f"cancelled {order_id}"


def my_llm(request: aa.LLMRequest) -> str:  # your model: any SDK, HTTP API or local model
    return "It is sunny in Oslo."


model = aa.CallableModel(my_llm, provider="local", model="my-llm")
executor = aa.Executor(registry)
```

### 1. Make a function a tool → `@registry.tool` (or `@aa.tool` on the default registry)

```python
@registry.tool(timeout=10, retries=2)  # read-only, so a retry is safe
def track_order(order_id: str) -> str:
    """Where an order is now."""
    return f"{order_id}: out for delivery"
```

### 2. Wrap code you cannot edit → `registry.wrap()`

```python
class OrderBook:  # a class from a library or legacy code
    def status(self, order_id: str) -> str:
        return "shipped"


registry.wrap(OrderBook().status, name="order_status", description="Order status.")
```

### 3. Keep a secret out of the model's view → `fixed=` on `registry.wrap()` (or read it from `ToolContext`)

```python
import os


def search_orders(query: str, api_key: str) -> str:
    """Search the order system."""
    return f"{query}: 2 orders"


secret = {"api_key": os.environ["ORDERS_API_KEY"]}  # unset: KeyError at start-up
registry.wrap(search_orders, fixed=secret)
```

### 4. Call a tool from sync or async code → `executor.run_sync()` / `await executor.run()`

```python
print(executor.run_sync("get_weather", {"city": "Oslo"}, timeout=30))
weather = await executor.run("get_weather", {"city": "Oslo"})  # in async code
```

### 5. Use one executor for the whole process → `aa.configure()` then `aa.call()` / `aa.acall()`

```python
aa.configure(registry, limits=aa.RunLimits(max_calls=50))
print(aa.call(get_weather, city="Oslo"))
```

### 6. Run an agent loop with no framework → `aa.run_tool_loop()`

```python
ask = [aa.Message(role="user", content="Weather in Oslo?")]
result = await aa.run_tool_loop(model, registry, ask, executor=executor)
print(result.final.text)
```

### 7. Make an agent a tool of another agent → `agent_tool(LoopAgent(...), executor=)` (`agent_adapter.exec`)

```python
from agent_adapter.exec import LoopAgent, agent_tool

desk = LoopAgent(model, registry.subset(names=["get_weather"]), instructions="Weather only.")
team = aa.Registry()
boss = aa.Executor(team)  # the desk's calls share its hooks, budget, approvals and run
team.add(agent_tool(desk, name="weather_desk", description="Weather.", executor=boss))
ask = [aa.Message(role="user", content="Oslo?")]
result = await aa.run_tool_loop(model, team, ask, executor=boss)  # nesting: max_depth=3
```

### 8. Get structured output (JSON matching a schema) → `output_schema=OutputSchema(...)`, then `.final.parsed`

```python
from agent_adapter.models import OutputSchema

shape = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
schema = OutputSchema(name="Answer", json_schema=shape)
result = await aa.run_tool_loop(model, registry, ask, executor=executor, output_schema=schema)
print(result.final.parsed["city"])
```

### 9. Fan out many calls in parallel → `executor.run_many()`

```python
cities = ("Oslo", "Rome", "Lima")
calls = [aa.ToolCall(name="get_weather", arguments={"city": c}) for c in cities]
results = await executor.run_many(calls, concurrency=4)  # one ToolResult per call
```

### 10. Fan in: steps that need earlier results → `executor.run_plan()` with `aa.Step(depends_on=)`

```python
first = aa.Step(id="a", tool="get_weather", args={"city": "Oslo"})
then = aa.Step(id="b", tool="get_weather", depends_on=["a"], args=lambda d: {"city": d["a"].value})
by_id = await executor.run_plan([first, then])
```

### 11. Run any async work in parallel with a deadline → `executor.parallel()`

```python
async def fetch(url: str) -> str:  # your own async work
    return url


pages = await executor.parallel(lambda: fetch("a"), lambda: fetch("b"), timeout=30)
```

### 12. Ask a person before a side effect → `confirm=True` on the tool plus an approval port on the executor

```python
def review(request: aa.ApprovalRequest) -> bool:  # ask in your UI, chat or CLI
    return request.arguments["order_id"].startswith("o-")


executor = aa.Executor(registry, approval=aa.CallbackApproval(review))
print(executor.run_sync("cancel_order", {"order_id": "o-1"}))
```

### 13. Park an approval and resume it later (any process) → `DeferredApproval`, `ApprovalPending`, `resume_sync()`

```python
executor = aa.Executor(registry, approval=aa.DeferredApproval.sqlite("approvals.db"))
try:
    executor.run_sync("cancel_order", {"order_id": "o-1"})
except aa.ApprovalPending as pending:  # store pending.approval_id for a reviewer
    # later, in any process that opens approvals.db:
    done = executor.resume_sync(pending.approval_id, True, decided_by="ana")
    print(done.value)  # the call runs once; a second resume is refused
```

### 14. Share approvals across processes and hosts → `SQLAlchemyApprovalStore` (or `aa.RedisApprovalStore`)

```python
from agent_adapter.approval import SQLAlchemyApprovalStore

# needs an async driver: pip install aiosqlite (one host), or asyncpg for
# "postgresql+asyncpg://..." (every host shares one database)
store = SQLAlchemyApprovalStore("sqlite+aiosqlite:///approvals.db")
executor = aa.Executor(registry, approval=aa.DeferredApproval(store))
```

### 15. Cap spend and tool calls per run → `aa.Ledger`, `aa.Prices`, `aa.BudgetHook`, `aa.RunLimits`

```python
mine = {"my-llm": {"provider": "local", "input": 1.0, "output": 4.0}}  # USD per 1M
ledger = aa.Ledger(prices=aa.Prices.load({"version": "mine", "models": mine}))
executor = aa.Executor(
    registry, ledger=ledger, hooks=[aa.BudgetHook(0.50)], limits=aa.RunLimits(max_calls=20)
)  # once_per_run=True too
```

### 16. Block, redact or approve personal data and secrets → `PIIGuard`, `CallableGuard(fn)` hooks (`agent_adapter.hooks`)

```python
from agent_adapter.hooks import CallableGuard, PIIGuard


def moderate(stage: str, payload: object) -> bool | str:  # or call your guard service
    return "off topic" if "lottery" in str(payload) else True


redact = PIIGuard(on=("result", "llm_response"), action="redact")  # emails, cards, keys
no_secrets = PIIGuard(kinds=["secret", "card"], on="args")  # or action="approve"
guards = [no_secrets, redact, CallableGuard(moderate, on=("args", "llm_request"))]
executor = aa.Executor(registry, hooks=guards)
print(executor.run_sync("get_weather", {"city": "Oslo"}))
```

### 17. Cache tool results → `cache_ttl=` / `cache_scope=` on the tool, `cache=` on the executor

```python
# global: every caller shares one entry; public data only, never per-user data
registry.wrap(get_weather, name="cached_weather", cache_ttl=60, cache_scope="global")
executor = aa.Executor(registry, cache=aa.MemoryCache())
# many processes: cache=agent_adapter.cache.RedisCache(url) (extra `redis`)
```

### 18. Cache model replies → `executor.model(model, cache_ttl=)`

```python
executor = aa.Executor(registry, cache=aa.MemoryCache())
llm = executor.model(model, cache_ttl=3600)  # also hooks, ledger and events
ask = aa.LLMRequest(messages=(aa.Message(role="user", content="hi"),))
first, again = await llm.generate(ask), await llm.generate(ask)  # again.cached
```

### 19. Log and trace every call → `aa.LoggingHook` and an event sink

```python
from agent_adapter.tracing import EventLog  # or OTelSink() (extra `otel`)

events = EventLog()
executor = aa.Executor(registry, hooks=[aa.LoggingHook()], sink=events)
executor.run_sync("get_weather", {"city": "Oslo"})  # secret-named keys are redacted
```

### 20. Bring your own model → `aa.CallableModel` (sync or async function)

```python
def house_llm(request: aa.LLMRequest) -> dict:  # call your SDK or HTTP API here
    return {"text": "Hello", "usage": {"input_tokens": 12, "output_tokens": 3}}


model = aa.CallableModel(house_llm, provider="house", model="house-13b")
```

### 21. Use a hosted provider or a local OpenAI-compatible server → `OpenAIModel` / `AnthropicModel` / `GeminiModel`

```python
from agent_adapter.models import AnthropicModel, OpenAIModel

model = OpenAIModel("gpt-4.1")  # reads OPENAI_API_KEY on the first call
claude = AnthropicModel("claude-sonnet-4-5")  # reads ANTHROPIC_API_KEY
ollama = {"base_url": "http://localhost:11434/v1", "api_key": "unused"}  # or vLLM
llama = OpenAIModel("llama3.1", api="chat", strict_tools=False, client_kwargs=ollama)
```

### 22. Fall back to a second model → `aa.FallbackModel`

```python
backup = aa.CallableModel(lambda request: "Try again later.", model="backup")
model = aa.FallbackModel([model, backup])  # on rate limit, outage or timeout
```

### 23. Bring your own embedder → `CallableEmbedder`

```python
from agent_adapter.embeddings import CallableEmbedder

embedder = CallableEmbedder(lambda texts: [[1.0, 0.5] for _ in texts], model="mine")
vectors = (await embedder.embed(["hello", "world"])).vectors
```

### 24. Price your own models → `aa.Prices.default().merged()` (or `aa.Prices.empty()`)

```python
mine = {
    "my-llm": {"provider": "local", "free": True},
    "gpu-*": {"provider": "rent", "per_second": 0.001},
}  # also per_call
prices = aa.Prices.default().merged({"version": "mine", "models": mine})
```

### 25. Compress a long context for one call → `CompressionPolicy`, `LLMRequest(compression=)`

```python
from agent_adapter.context import CompressionPolicy, RemoveOldMessages

policy = CompressionPolicy(
    start_when_tokens_over=8_000, shrink_to_tokens=6_000, steps=[RemoveOldMessages(keep_last=6)]
)
reply = await model.generate(aa.LLMRequest(messages=history, compression=policy))
```

### 26. Compress messages in a LangGraph node → `langgraph_compressor`

```python
from agent_adapter.integrations.langgraph import langgraph_compressor

compressor = langgraph_compressor(policy)  # `policy` as in the recipe above
messages = await compressor(state["messages"])  # in your node, before the model
```

### 27. Compress any framework's message list → `compress_for()` / `compress_for_sync()`

```python
from agent_adapter.context import compress_for_sync

smaller, report = compress_for_sync(chat_dicts, policy, format="openai")
print(report.tokens_before, report.tokens_after)  # `smaller` is still chat dicts
```

### 28. Attach tools to any framework → `aa.attach()` (by name or by framework object)

```python
tools = aa.attach("openai", registry, executor=executor)  # Chat Completions tools
# aa.attach(agent_or_graph, registry, executor=executor): LangGraph, Agents SDK...
```

### 29. Import tools from an OpenAPI document (or an MCP server) → `import_openapi` (or `import_mcp`)

```python
from agent_adapter.integrations.openapi import import_openapi

api = {"spec": "openapi.json", "base_url": "https://api.example.com"}
import_openapi(registry, **api, operations=["getForecast"])  # reads: no approval
# writes and deletes run only after an approval (confirm=True):
import_openapi(registry, **api, operations=["cancelBooking"], confirm=True)
```

### 30. Serve tools from FastAPI → `await executor.run()` in an async route

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/weather/{city}")
async def weather(city: str) -> str:
    return await executor.run("get_weather", {"city": city})
```

### 31. Serve tools from Flask → `executor.run_sync()` in a view

```python
from flask import Flask

app = Flask(__name__)


@app.get("/weather/<city>")
def weather(city: str) -> str:
    return executor.run_sync("get_weather", {"city": city}, timeout=30)
```

### 32. Serve tools from Django → `await executor.run()` in an async view (`run_sync` in a sync one)

```python
from django.http import JsonResponse


async def weather(request, city: str) -> JsonResponse:
    return JsonResponse({"text": await executor.run("get_weather", {"city": city})})
```

### 33. Run tools in a Celery task → `executor.run_sync()` in the task

```python
from celery import Celery

app = Celery("tasks")


@app.task
def weather(city: str) -> str:
    return executor.run_sync("get_weather", {"city": city}, timeout=30)
```

### 34. Run tools in AWS Lambda (SQS, EventBridge, direct) → `tool_handler`

```python
from agent_adapter.handlers import tool_handler

handler = tool_handler(executor, tools={"get_weather"}, timeout=30)  # entry point
```

### 35. Run tools from any queue (RabbitMQ, Kafka, Pub/Sub) → `handle_envelope`

```python
from agent_adapter.handlers import handle_envelope

body = {"tool": "get_weather", "arguments": {"city": "Oslo"}}  # the message body
reply = await handle_envelope(executor, body, tools={"get_weather"}, message_id="m1")
```

### 36. Build a flow (steps, branches, fan-out, pauses) → `aa.Flow`, `aa.FlowRunner`, `Call` / `Route` / `Map` / `Join`

```python
from pydantic import BaseModel
from agent_adapter.flow import Call


class Trip(BaseModel):
    city: str
    weather: str = ""


look = Call("get_weather", args=lambda trip: {"city": trip.city}, into="weather")
flow = aa.Flow("trip", state=Trip).node("look", look).end("look")
result = await aa.FlowRunner(executor).run(flow, {"city": "Oslo"})
```

### 37. Serve tools over MCP → `streamable_http_app`

```python
from agent_adapter.integrations.mcp import streamable_http_app

app = streamable_http_app(registry, executor=executor)  # ASGI app at /mcp
# no auth= + token_verifier=: anonymous, so it serves loopback clients only
```

### 38. Serve tools over HTTP for other languages → `create_app`

```python
from agent_adapter.server import Principal, bearer_auth, create_app

auth = bearer_auth({"s3cret": Principal(id="web-app")})  # tokens from a vault
app = create_app(registry, executor=executor, authenticate=auth)  # uvicorn mod:app
```

### 39. Serve tools as an A2A agent → `a2a_app`

```python
from agent_adapter.integrations.a2a import a2a_app, static_bearer

auth = static_bearer({"s3cret": "agent-team"})  # token -> caller id; from a vault
app = a2a_app(
    registry,
    url="https://tools.example.com/a2a/",
    executor=executor,
    authenticate=auth,
    host="0.0.0.0",
    allowed_hosts=["tools.example.com"],
)
```

### 40. Improve tool descriptions and prompts against eval cases → `CallableHost`, `optimise` (or the optimiser CLI)

```python
from agent_adapter.optimiser import CallableHost, OptimiserConfig, load_cases, optimise

cases = load_cases("cases.jsonl")  # one {'id', 'input', 'expected'} per line
host = CallableHost(run_case, surface={"prompt/system": "Be brief."}, cases=cases)
result = await optimise(host, OptimiserConfig(), proposer=model, model=model)
print(result.patch)  # a reviewable diff; nothing is applied
```

### 41. See runs in a browser (timeline, flow graph, approvals) → `agent_adapter.viewer.record()`, then `agent-adapter ui`

```python
from agent_adapter.viewer import record

record(executor)  # every run goes to .agent_adapter/runs
# then, in a terminal: agent-adapter ui
```

### 42. Chat with any agent or model in a local page → `viewer.serve(...)` or `agent-adapter ui --agent`

```python
from agent_adapter.viewer import ChatMessage, ChatSession, serve


async def chat(message: ChatMessage, session: ChatSession) -> str:
    return f"you said: {message.text}"  # your agent; tools: ctx=session.context()


if __name__ == "__main__":
    serve(agent=chat)  # or serve(model=model, registry=registry)
# CLI: agent-adapter ui --agent app:chat
```

### More recipes: code in the full guide (`agent-adapter guide --full`)

- Hand the conversation to another agent
- Send spans to Langfuse, Phoenix or any OTLP backend
- Wrap a vendor SDK (Bedrock, Azure OpenAI, Mistral, Groq, OpenRouter)
- Use a known model's context window
- Remember facts per user across chats (recall, forget, export, erase)
- Let the model search your documents and cite sources (RAG)
- Remember a chat user across chats (recall each turn, save facts or a summary)
- Show reviewers similar past approvals; propose confirm rules a person accepts
- Run a flow inside a flow, or cache a node
- Evaluate an agent on eval cases (scores, JUnit for CI, regressions)

More features (full recipes in llms-full.txt): durable runs, Temporal and DBOS, sandboxed code, team budgets, streaming deltas, viewer search, tags and OTLP import, framework templates and migrate, policy file, audit log, evals from production, cost advisor.

## Common errors → fixes

Every error subclasses `AdapterError` and carries a stable `kind`.

| Error | Fix |
|---|---|
| `AdapterError` | Catch it to handle any adapter failure; read `kind` to tell them apart |
| `ToolDefinitionError` | Read the message: a valid name (letters, digits, `_`), a description or docstring, a signature it can model |
| `DuplicateTool` | Pick another `name=`, or use a separate `Registry` |
| `ToolNotFound` | Check the name and the registry; the model may have invented a tool |
| `ToolArgsInvalid` | Send `.errors` back to the model so it can retry with valid arguments |
| `ToolUsageError` | Fix the host code the message names: a confirm rule, an approval callback, a reused call id |
| `SchemaNotPortable` | Read `.problems`; simplify the type (see Schemas) |
| `MissingDependency` | Install the extra the message names |
| `ToolError` | Raise it from your tool with a message the model can act on |
| `ToolTimeout` | Retry, raise the tool's `timeout`, or make the tool faster |
| `ToolFailed` | Read `cause_type` and `__cause__`, fix the tool, or raise `ToolError` for expected failures |
| `RunLimitExceeded` | Start a new run, or raise the tool's run limit |
| `GuardrailBlocked` | Read `.reason` and `.stage`; change the input or the guardrail check |
| `ApprovalDenied` | Tell the model the call was refused; do not retry it unchanged |
| `ApprovalPending` | Keep `.approval_id`, ask a reviewer, then resume with the decision |
| `ApprovalNotFound` | Check the id; it may have expired or belong to another store |
| `ApprovalAlreadyDecided` | Read the stored decision instead of deciding again |
| `ApprovalStoreCorrupt` | Treat it as tampering: inspect the row, then delete it and ask again (Approval stores) |
| `BudgetExceeded` | Stop the run, or raise the `BudgetHook` limit |
| `PlanInvalid` | Fix each item in `.problems` |
| `FlowError` | Read `.flow` and `.node`; the cause is in `__cause__` |
| `FlowDefinitionError` | Fix each item in `.problems` |
| `FlowCompileError` | Drop what `.target` does not support, or run the flow with `FlowRunner` |
| `FlowLoopExceeded` | Fix the route that loops, or raise the node's loop limit |
| `CheckpointNotFound` | Check the run id and that the runner uses the store the run was saved in |
| `CheckpointConflict` | Leave the run to the worker that holds it; a run that ended needs no recovery (Durable runs) |
| `CheckpointStoreCorrupt` | Check `checkpoint_key` is the one the run was saved with; else treat it as tampering and delete the run |
| `RecoveryNeedsDecision` | Check each call in `.calls` at its system, then recover with `rerun_in_doubt=True`, or mark the tool `idempotent=True` |
| `ConfigError` | Fix the setting the message names |
| `IntegrationError` | Check the integration name and the framework version |
| `AuthError` | Catch it to handle both subclasses below |
| `Unauthorized` | Send a valid bearer token |
| `Forbidden` | Grant the principal the scope it needs, such as the approver scope |
| `OptimiserError` | Read the message; check the eval cases and the optimiser budget |
| `ModelError` | Catch it for any provider failure; check `retryable` before retrying |
| `ModelRateLimited` | Wait `.retry_after` seconds, then retry |
| `ModelUnavailable` | Retry with backoff, or fall back with `FallbackModel` |
| `ModelTimeout` | Retry, or raise the client timeout |
| `ModelAuthError` | Check the API key and its permissions |
| `ModelBadRequest` | Read the message; check the model name, tool schemas and strict mode |
| `ModelContextOverflow` | Send fewer tokens: a `CompressionPolicy` with `if_still_too_long` retries once (Context compression) |
| `ModelContentFiltered` | Change the prompt or input; do not retry it unchanged |
| `ModelOutputInvalid` | Retry with a clearer prompt, or loosen the output schema |

Symptoms and their fixes: <https://agent-adapter.veloceai.in/troubleshooting/>.

## CLI commands

- `agent-adapter init fastapi my-service`: write a starter project from a tested template
- `agent-adapter doctor`: check Python, extras, optional backends, integrations and the pip index
- `agent-adapter guide --out docs/agent-adapter.md`: print the guide for AI coding agents (AGENTS.md)
- `agent-adapter ui`: browse recorded runs, or chat with an agent, in a local web page
- `agent-adapter eval app.chat:agent eval_cases.jsonl --min-score 0.9 --junit evals.xml`: run eval cases on an agent, grade them, and fail below a score
- `agent-adapter migrate src/ --diff`: list framework tool definitions and suggest registry code (read-only)
- `agent-adapter policy check policy.yaml --env prod`: validate a policy file and print the rules it sets
- `agent-adapter audit verify .agent_adapter/audit`: verify a signed audit log, or export it as JSONL or CSV
- `agent-adapter eval app:agent cases.jsonl --save`: also save the report; `agent-adapter ui` shows it under Evals (`--evals FOLDER`)
- `agent-adapter guide --full`: the guide plus every public signature
- `python -m agent_adapter guide --section recipes`: the same CLI without the script
- `python -m agent_adapter.server --registry mypkg.tools:registry`: serve a registry over HTTP (tokens from `AGENT_ADAPTER_TOKENS`)
- `python -m agent_adapter.optimiser --host mypkg.evals:build_host --dry-run`: plan an optimiser run without calling a model

## Full API index

Every public name with its signature and first docstring line, by module:
`agent-adapter guide --section api`, `agent-adapter guide --full`, or
<https://agent-adapter.veloceai.in/llms-full.txt>.
