"""Recipes for tools, running, agents, guards, approvals, budgets, cache, tracing, models."""

from __future__ import annotations

from scripts.agent_guide.recipe_model import Recipe

TOOLS: tuple[Recipe, ...] = (
    Recipe(
        "make a function a tool",
        "`@registry.tool` (or `@aa.tool` on the default registry)",
        (
            "@registry.tool(timeout=10, retries=2)  # read-only, so a retry is safe",
            "def track_order(order_id: str) -> str:",
            '    """Where an order is now."""',
            '    return f"{order_id}: out for delivery"',
        ),
    ),
    Recipe(
        "wrap code you cannot edit",
        "`registry.wrap()`",
        (
            "class OrderBook:  # a class from a library or legacy code",
            "    def status(self, order_id: str) -> str:",
            '        return "shipped"',
            "",
            "",
            'registry.wrap(OrderBook().status, name="order_status", description="Order status.")',
        ),
    ),
    Recipe(
        "keep a secret out of the model's view",
        "`fixed=` on `registry.wrap()` (or read it from `ToolContext`)",
        (
            "import os",
            "",
            "",
            "def search_orders(query: str, api_key: str) -> str:",
            '    """Search the order system."""',
            '    return f"{query}: 2 orders"',
            "",
            "",
            'secret = {"api_key": os.environ["ORDERS_API_KEY"]}  # unset: KeyError at start-up',
            "registry.wrap(search_orders, fixed=secret)",
        ),
        env=(("ORDERS_API_KEY", "test-key"),),
    ),
    Recipe(
        "call a tool from sync or async code",
        "`executor.run_sync()` / `await executor.run()`",
        (
            'print(executor.run_sync("get_weather", {"city": "Oslo"}, timeout=30))',
            'weather = await executor.run("get_weather", {"city": "Oslo"})  # in async code',
        ),
    ),
    Recipe(
        "use one executor for the whole process",
        "`aa.configure()` then `aa.call()` / `aa.acall()`",
        (
            "aa.configure(registry, limits=aa.RunLimits(max_calls=50))",
            'print(aa.call(get_weather, city="Oslo"))',
        ),
    ),
    Recipe(
        "run an agent loop with no framework",
        "`aa.run_tool_loop()`",
        (
            'ask = [aa.Message(role="user", content="Weather in Oslo?")]',
            "result = await aa.run_tool_loop(model, registry, ask, executor=executor)",
            "print(result.final.text)",
        ),
    ),
    Recipe(
        "make an agent a tool of another agent",
        "`agent_tool(LoopAgent(...), executor=)` (`agent_adapter.exec`)",
        (
            "from agent_adapter.exec import LoopAgent, agent_tool",
            "",
            'desk = LoopAgent(model, registry.subset(names=["get_weather"]), '
            'instructions="Weather only.")',
            "team = aa.Registry()",
            "boss = aa.Executor(team)  # the desk's calls share its hooks, budget, approvals and "
            "run",
            'team.add(agent_tool(desk, name="weather_desk", description="Weather.", '
            "executor=boss))",
            'ask = [aa.Message(role="user", content="Oslo?")]',
            "result = await aa.run_tool_loop(model, team, ask, executor=boss)  # nesting: "
            "max_depth=3",
        ),
    ),
    Recipe(
        "hand the conversation to another agent",
        "`handoff(...)` + `HandoffModel(model, registry)`",
        (
            "from agent_adapter.exec import HandoffModel, LoopAgent, handoff",
            "",
            'desk = LoopAgent(model, registry.subset(names=["get_weather"]))',
            "front = aa.Registry()",
            'front.add(handoff(desk, name="to_weather", description="Weather desk.", '
            "executor=executor))",
            'ask = [aa.Message(role="user", content="Oslo?")]',
            "front_desk = HandoffModel(model, front)  # the desk's answer ends the loop",
            "result = await aa.run_tool_loop(front_desk, front, ask, executor=executor)",
        ),
        full_only=True,
    ),
    Recipe(
        "get structured output (JSON matching a schema)",
        "`output_schema=OutputSchema(...)`, then `.final.parsed`",
        (
            "from agent_adapter.models import OutputSchema",
            "",
            'shape = {"type": "object", "properties": {"city": {"type": "string"}}, '
            '"required": ["city"]}',
            'schema = OutputSchema(name="Answer", json_schema=shape)',
            "result = await aa.run_tool_loop("
            "model, registry, ask, executor=executor, output_schema=schema)",
            'print(result.final.parsed["city"])',
        ),
        setup=(
            'model = aa.CallableModel(lambda request: \'{"city": "Oslo"}\', model="json")\n'
            'ask = [aa.Message(role="user", content="Which city?")]'
        ),
    ),
    Recipe(
        "fan out many calls in parallel",
        "`executor.run_many()`",
        (
            'cities = ("Oslo", "Rome", "Lima")',
            'calls = [aa.ToolCall(name="get_weather", arguments={"city": c}) for c in cities]',
            "results = await executor.run_many(calls, concurrency=4)  # one ToolResult per call",
        ),
    ),
    Recipe(
        "fan in: steps that need earlier results",
        "`executor.run_plan()` with `aa.Step(depends_on=)`",
        (
            'first = aa.Step(id="a", tool="get_weather", args={"city": "Oslo"})',
            'then = aa.Step(id="b", tool="get_weather", depends_on=["a"], '
            'args=lambda d: {"city": d["a"].value})',
            "by_id = await executor.run_plan([first, then])",
        ),
    ),
    Recipe(
        "run any async work in parallel with a deadline",
        "`executor.parallel()`",
        (
            "async def fetch(url: str) -> str:  # your own async work",
            "    return url",
            "",
            "",
            'pages = await executor.parallel(lambda: fetch("a"), lambda: fetch("b"), timeout=30)',
        ),
    ),
    Recipe(
        "ask a person before a side effect",
        "`confirm=True` on the tool plus an approval port on the executor",
        (
            "def review(request: aa.ApprovalRequest) -> bool:  # ask in your UI, chat or CLI",
            '    return request.arguments["order_id"].startswith("o-")',
            "",
            "",
            "executor = aa.Executor(registry, approval=aa.CallbackApproval(review))",
            'print(executor.run_sync("cancel_order", {"order_id": "o-1"}))',
        ),
    ),
    Recipe(
        "park an approval and resume it later (any process)",
        "`DeferredApproval`, `ApprovalPending`, `resume_sync()`",
        (
            'executor = aa.Executor(registry, approval=aa.DeferredApproval.sqlite("approvals.db"))',
            "try:",
            '    executor.run_sync("cancel_order", {"order_id": "o-1"})',
            "except aa.ApprovalPending as pending:  # store pending.approval_id for a reviewer",
            "    # later, in any process that opens approvals.db:",
            '    done = executor.resume_sync(pending.approval_id, True, decided_by="ana")',
            "    print(done.value)  # the call runs once; a second resume is refused",
        ),
    ),
    Recipe(
        "share approvals across processes and hosts",
        "`SQLAlchemyApprovalStore` (or `aa.RedisApprovalStore`)",
        (
            "from agent_adapter.approval import SQLAlchemyApprovalStore",
            "",
            "# needs an async driver: pip install aiosqlite (one host), or asyncpg for",
            '# "postgresql+asyncpg://..." (every host shares one database)',
            'store = SQLAlchemyApprovalStore("sqlite+aiosqlite:///approvals.db")',
            "executor = aa.Executor(registry, approval=aa.DeferredApproval(store))",
        ),
        requires=("sqlalchemy", "aiosqlite"),
    ),
    Recipe(
        "cap spend and tool calls per run",
        "`aa.Ledger`, `aa.Prices`, `aa.BudgetHook`, `aa.RunLimits`",
        (
            'mine = {"my-llm": {"provider": "local", "input": 1.0, "output": 4.0}}  # USD per 1M',
            'ledger = aa.Ledger(prices=aa.Prices.load({"version": "mine", "models": mine}))',
            "executor = aa.Executor(",
            "    registry, ledger=ledger, hooks=[aa.BudgetHook(0.50)], "
            "limits=aa.RunLimits(max_calls=20)",
            ")  # once_per_run=True too",
        ),
    ),
    Recipe(
        "block, redact or approve personal data and secrets",
        "`PIIGuard`, `CallableGuard(fn)` hooks (`agent_adapter.hooks`)",
        (
            "from agent_adapter.hooks import CallableGuard, PIIGuard",
            "",
            "",
            "def moderate(stage: str, payload: object) -> bool | str:  # or call your guard "
            "service",
            '    return "off topic" if "lottery" in str(payload) else True',
            "",
            "",
            'redact = PIIGuard(on=("result", "llm_response"), action="redact")  # emails, cards, '
            "keys",
            'no_secrets = PIIGuard(kinds=["secret", "card"], on="args")  # or action="approve"',
            'guards = [no_secrets, redact, CallableGuard(moderate, on=("args", "llm_request"))]',
            "executor = aa.Executor(registry, hooks=guards)",
            'print(executor.run_sync("get_weather", {"city": "Oslo"}))',
        ),
    ),
    Recipe(
        "cache tool results",
        "`cache_ttl=` / `cache_scope=` on the tool, `cache=` on the executor",
        (
            "# global: every caller shares one entry; public data only, never per-user data",
            'registry.wrap(get_weather, name="cached_weather", cache_ttl=60, cache_scope="global")',
            "executor = aa.Executor(registry, cache=aa.MemoryCache())",
            "# many processes: cache=agent_adapter.cache.RedisCache(url) (extra `redis`)",
        ),
    ),
    Recipe(
        "cache model replies",
        "`executor.model(model, cache_ttl=)`",
        (
            "executor = aa.Executor(registry, cache=aa.MemoryCache())",
            "llm = executor.model(model, cache_ttl=3600)  # also hooks, ledger and events",
            'ask = aa.LLMRequest(messages=(aa.Message(role="user", content="hi"),))',
            "first, again = await llm.generate(ask), await llm.generate(ask)  # again.cached",
        ),
    ),
    Recipe(
        "log and trace every call",
        "`aa.LoggingHook` and an event sink",
        (
            "from agent_adapter.tracing import EventLog  # or OTelSink() (extra `otel`)",
            "",
            "events = EventLog()",
            "executor = aa.Executor(registry, hooks=[aa.LoggingHook()], sink=events)",
            'executor.run_sync("get_weather", {"city": "Oslo"})  # secret-named keys are redacted',
        ),
    ),
    Recipe(
        "send spans to Langfuse, Phoenix or any OTLP backend",
        "`OTelSink(tracer_provider=)` with an OTLP exporter",
        (
            "from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter",
            "from opentelemetry.sdk.trace import TracerProvider",
            "from opentelemetry.sdk.trace.export import BatchSpanProcessor",
            "from agent_adapter.tracing import OTelSink",
            "",
            'endpoint = "http://localhost:6006/v1/traces"  # Phoenix; Langfuse: its URL and auth '
            "headers",
            "provider = TracerProvider()",
            "provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))",
            "executor = aa.Executor(registry, sink=OTelSink(tracer_provider=provider))",
        ),
        requires=("opentelemetry.sdk", "opentelemetry.exporter.otlp"),
        full_only=True,
    ),
    Recipe(
        "bring your own model",
        "`aa.CallableModel` (sync or async function)",
        (
            "def house_llm(request: aa.LLMRequest) -> dict:  # call your SDK or HTTP API here",
            '    return {"text": "Hello", "usage": {"input_tokens": 12, "output_tokens": 3}}',
            "",
            "",
            'model = aa.CallableModel(house_llm, provider="house", model="house-13b")',
        ),
    ),
    Recipe(
        "wrap a vendor SDK (Bedrock, Azure OpenAI, Mistral, Groq, OpenRouter)",
        "`CallableModel` + `build_chat_request` / `parse_chat_completion`",
        (
            "from agent_adapter.models.openai import build_chat_request, parse_chat_completion",
            "",
            "",
            "def vendor_llm(request: aa.LLMRequest) -> dict:  # any OpenAI-shaped SDK",
            '    kwargs = build_chat_request(request, model="llama-3.3-70b", strict_tools=False)',
            "    reply = parse_chat_completion(client.chat.completions.create(**kwargs))",
            '    return {"text": reply.text, "tool_calls": reply.tool_calls, "usage": reply.usage}',
            "",
            "",
            'model = aa.CallableModel(vendor_llm, provider="groq", model="llama-3.3-70b")',
            'answer = await model.generate(aa.LLMRequest(messages=(aa.Message(role="user", '
            'content="hi"),)))',
        ),
        setup=(
            "from types import SimpleNamespace\n"
            'done = {"id": "c1", "model": "llama-3.3-70b", "choices": [{"index": 0, '
            '"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}], '
            '"usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}\n'
            "create = lambda **kwargs: done\n"
            "client = "
            "SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))"
        ),
        full_only=True,
    ),
    Recipe(
        "use a hosted provider or a local OpenAI-compatible server",
        "`OpenAIModel` / `AnthropicModel` / `GeminiModel`",
        (
            "from agent_adapter.models import AnthropicModel, OpenAIModel",
            "",
            'model = OpenAIModel("gpt-4.1")  # reads OPENAI_API_KEY on the first call',
            'claude = AnthropicModel("claude-sonnet-4-5")  # reads ANTHROPIC_API_KEY',
            'ollama = {"base_url": "http://localhost:11434/v1", "api_key": "unused"}  # or vLLM',
            'llama = OpenAIModel("llama3.1", api="chat", strict_tools=False, client_kwargs=ollama)',
        ),
        requires=("openai", "anthropic"),
    ),
    Recipe(
        "fall back to a second model",
        "`aa.FallbackModel`",
        (
            'backup = aa.CallableModel(lambda request: "Try again later.", model="backup")',
            "model = aa.FallbackModel([model, backup])  # on rate limit, outage or timeout",
        ),
    ),
    Recipe(
        "bring your own embedder",
        "`CallableEmbedder`",
        (
            "from agent_adapter.embeddings import CallableEmbedder",
            "",
            'embedder = CallableEmbedder(lambda texts: [[1.0, 0.5] for _ in texts], model="mine")',
            'vectors = (await embedder.embed(["hello", "world"])).vectors',
        ),
    ),
    Recipe(
        "price your own models",
        "`aa.Prices.default().merged()` (or `aa.Prices.empty()`)",
        (
            "mine = {",
            '    "my-llm": {"provider": "local", "free": True},',
            '    "gpu-*": {"provider": "rent", "per_second": 0.001},',
            "}  # also per_call",
            'prices = aa.Prices.default().merged({"version": "mine", "models": mine})',
        ),
    ),
    Recipe(
        "use a known model's context window",
        "`aa.Prices.default().caps(model)`",
        (
            'caps = aa.Prices.default().caps("gpt-5.4")',
            "print(caps.context_window, caps.prompt_window())  # percent compression needs no "
            "setting",
        ),
        full_only=True,
    ),
)
