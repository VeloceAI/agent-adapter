"""Recipes for compression, frameworks, backends, queues, flows, servers and the optimiser."""

from __future__ import annotations

from scripts.agent_guide.recipe_model import POLICY, Recipe

# Test-only inputs: a one-operation OpenAPI document and a one-case eval file.
OPENAPI = (
    "import json\n"
    'op = {"operationId": "getForecast", "summary": "Forecast for a city.",\n'
    '      "parameters": [{"name": "city", "in": "path", "required": True,\n'
    '                      "schema": {"type": "string"}}],\n'
    '      "responses": {"200": {"description": "ok"}}}\n'
    'cancel = {"operationId": "cancelBooking", "summary": "Cancel a booking.",\n'
    '          "parameters": [{"name": "id", "in": "path", "required": True,\n'
    '                          "schema": {"type": "string"}}],\n'
    '          "responses": {"204": {"description": "cancelled"}}}\n'
    'spec = {"openapi": "3.1.0", "info": {"title": "Weather", "version": "1"},\n'
    '        "paths": {"/forecast/{city}": {"get": op}, "/bookings/{id}": {"delete": cancel}}}\n'
    "from pathlib import Path\n"
    'Path("openapi.json").write_text(json.dumps(spec), encoding="utf-8")'
)
CASES = (
    "from pathlib import Path\n"
    'Path("cases.jsonl").write_text(\'{"id": "c1", "input": "Weather in Oslo?"}\', "utf-8")\n'
    "async def run_case(tree, case, model):  # your agent, one case\n"
    "    return 'sunny'"
)

EVAL_SETUP = (
    "from pathlib import Path\n"
    'Path("cases.jsonl").write_text(\'{"id": "c1", "input": "Weather in Oslo?", '
    '"expected": "sunny"}\', "utf-8")\n'
    "def answer(question):  # your agent: fn(input), fn(input, ctx=) or fn(message, session)\n"
    "    return 'sunny'"
)

HOSTS: tuple[Recipe, ...] = (
    Recipe(
        "compress a long context for one call",
        "`CompressionPolicy`, `LLMRequest(compression=)`",
        (
            "from agent_adapter.context import CompressionPolicy, RemoveOldMessages",
            "",
            "policy = CompressionPolicy(",
            "    start_when_tokens_over=8_000, shrink_to_tokens=6_000, "
            "steps=[RemoveOldMessages(keep_last=6)]",
            ")",
            "reply = await model.generate(aa.LLMRequest(messages=history, compression=policy))",
        ),
        setup='history = tuple(aa.Message(role="user", content="hi") for _ in range(3))',
    ),
    Recipe(
        "compress messages in a LangGraph node",
        "`langgraph_compressor`",
        (
            "from agent_adapter.integrations.langgraph import langgraph_compressor",
            "",
            "compressor = langgraph_compressor(policy)  # `policy` as in the recipe above",
            'messages = await compressor(state["messages"])  # in your node, before the model',
        ),
        requires=("langchain_core",),
        setup=POLICY + "from langchain_core.messages import HumanMessage\n"
        'state = {"messages": [HumanMessage(content="hi")]}',
    ),
    Recipe(
        "compress any framework's message list",
        "`compress_for()` / `compress_for_sync()`",
        (
            "from agent_adapter.context import compress_for_sync",
            "",
            'smaller, report = compress_for_sync(chat_dicts, policy, format="openai")',
            "print(report.tokens_before, report.tokens_after)  # `smaller` is still chat dicts",
        ),
        setup=POLICY + 'chat_dicts = [{"role": "user", "content": "hi"}]',
    ),
    Recipe(
        "attach tools to any framework",
        "`aa.attach()` (by name or by framework object)",
        (
            'tools = aa.attach("openai", registry, executor=executor)  # Chat Completions tools',
            "# aa.attach(agent_or_graph, registry, executor=executor): LangGraph, Agents SDK...",
        ),
    ),
    Recipe(
        "import tools from an OpenAPI document (or an MCP server)",
        "`import_openapi` (or `import_mcp`)",
        (
            "from agent_adapter.integrations.openapi import import_openapi",
            "",
            'api = {"spec": "openapi.json", "base_url": "https://api.example.com"}',
            'import_openapi(registry, **api, operations=["getForecast"])  # reads: no approval',
            "# writes and deletes run only after an approval (confirm=True):",
            'import_openapi(registry, **api, operations=["cancelBooking"], confirm=True)',
        ),
        requires=("httpx", "yaml"),
        setup=OPENAPI,
    ),
    Recipe(
        "serve tools from FastAPI",
        "`await executor.run()` in an async route",
        (
            "from fastapi import FastAPI",
            "",
            "app = FastAPI()",
            "",
            "",
            '@app.get("/weather/{city}")',
            "async def weather(city: str) -> str:",
            '    return await executor.run("get_weather", {"city": city})',
        ),
        requires=("fastapi",),
    ),
    Recipe(
        "serve tools from Flask",
        "`executor.run_sync()` in a view",
        (
            "from flask import Flask",
            "",
            "app = Flask(__name__)",
            "",
            "",
            '@app.get("/weather/<city>")',
            "def weather(city: str) -> str:",
            '    return executor.run_sync("get_weather", {"city": city}, timeout=30)',
        ),
        requires=("flask",),
    ),
    Recipe(
        "serve tools from Django",
        "`await executor.run()` in an async view (`run_sync` in a sync one)",
        (
            "from django.http import JsonResponse",
            "",
            "",
            "async def weather(request, city: str) -> JsonResponse:",
            '    return JsonResponse({"text": await executor.run("get_weather", {"city": city})})',
        ),
        requires=("django",),
    ),
    Recipe(
        "run tools in a Celery task",
        "`executor.run_sync()` in the task",
        (
            "from celery import Celery",
            "",
            'app = Celery("tasks")',
            "",
            "",
            "@app.task",
            "def weather(city: str) -> str:",
            '    return executor.run_sync("get_weather", {"city": city}, timeout=30)',
        ),
        requires=("celery",),
    ),
    Recipe(
        "run tools in AWS Lambda (SQS, EventBridge, direct)",
        "`tool_handler`",
        (
            "from agent_adapter.handlers import tool_handler",
            "",
            'handler = tool_handler(executor, tools={"get_weather"}, timeout=30)  # entry point',
        ),
    ),
    Recipe(
        "run tools from any queue (RabbitMQ, Kafka, Pub/Sub)",
        "`handle_envelope`",
        (
            "from agent_adapter.handlers import handle_envelope",
            "",
            'body = {"tool": "get_weather", "arguments": {"city": "Oslo"}}  # the message body',
            'reply = await handle_envelope(executor, body, tools={"get_weather"}, message_id="m1")',
        ),
    ),
    Recipe(
        "build a flow (steps, branches, fan-out, pauses)",
        "`aa.Flow`, `aa.FlowRunner`, `Call` / `Route` / `Map` / `Join`",
        (
            "from pydantic import BaseModel",
            "from agent_adapter.flow import Call",
            "",
            "",
            "class Trip(BaseModel):",
            "    city: str",
            '    weather: str = ""',
            "",
            "",
            'look = Call("get_weather", args=lambda trip: {"city": trip.city}, into="weather")',
            'flow = aa.Flow("trip", state=Trip).node("look", look).end("look")',
            'result = await aa.FlowRunner(executor).run(flow, {"city": "Oslo"})',
        ),
    ),
    Recipe(
        "run a flow inside a flow, or cache a node",
        "`SubFlow(flow, input=, into=)`, `Flow.node(cache=NodeCache(...))`",
        (
            "from pydantic import BaseModel",
            "from agent_adapter.flow import NodeCache, SubFlow",
            "",
            "",
            "class Trip(BaseModel):",
            "    city: str",
            '    weather: str = ""',
            "",
            "",
            'check = aa.Flow("check", state=Trip).node("look", lambda t: {"weather": '
            '"sun"}).end("look")',
            'cached = NodeCache(ttl=600, reads=["city"])  # needs a cache port on the executor',
            'flow = aa.Flow("trip", state=Trip).node("plan", SubFlow(check), '
            'cache=cached).end("plan")',
            "executor = aa.Executor(registry, cache=aa.MemoryCache())",
            'result = await aa.FlowRunner(executor).run(flow, {"city": "Oslo"})',
        ),
        full_only=True,
    ),
    Recipe(
        "serve tools over MCP",
        "`streamable_http_app`",
        (
            "from agent_adapter.integrations.mcp import streamable_http_app",
            "",
            "app = streamable_http_app(registry, executor=executor)  # ASGI app at /mcp",
            "# no auth= + token_verifier=: anonymous, so it serves loopback clients only",
        ),
        requires=("mcp",),
    ),
    Recipe(
        "serve tools over HTTP for other languages",
        "`create_app`",
        (
            "from agent_adapter.server import Principal, bearer_auth, create_app",
            "",
            'auth = bearer_auth({"s3cret": Principal(id="web-app")})  # tokens from a vault',
            "app = create_app(registry, executor=executor, authenticate=auth)  # uvicorn mod:app",
        ),
        requires=("starlette",),
    ),
    Recipe(
        "serve tools as an A2A agent",
        "`a2a_app`",
        (
            "from agent_adapter.integrations.a2a import a2a_app, static_bearer",
            "",
            'auth = static_bearer({"s3cret": "agent-team"})  # token -> caller id; from a vault',
            "app = a2a_app(",
            "    registry,",
            '    url="https://tools.example.com/a2a/",',
            "    executor=executor,",
            "    authenticate=auth,",
            '    host="0.0.0.0",',
            '    allowed_hosts=["tools.example.com"],',
            ")",
        ),
        requires=("a2a",),
    ),
    Recipe(
        "evaluate an agent on eval cases (scores, JUnit for CI, regressions)",
        "`run_eval_sync` with graders (or `agent-adapter eval TARGET CASES`)",
        (
            "from agent_adapter.evals import load_cases, run_eval_sync",
            "from agent_adapter.evals.graders import contains, max_cost",
            "",
            'cases, graders = load_cases("cases.jsonl"), [contains(), max_cost(0.05)]',
            'report = run_eval_sync(answer, cases, graders, save_to=".agent_adapter/evals")',
            "print(report.mean_score, report.n_passed)  # then: agent-adapter ui, Evals tab",
            'Path("evals.xml").write_text(report.to_junit(), "utf-8")  # for CI test reports',
        ),
        setup=EVAL_SETUP,
        full_only=True,
    ),
    Recipe(
        "improve tool descriptions and prompts against eval cases",
        "`CallableHost`, `optimise` (or the optimiser CLI)",
        (
            "from agent_adapter.optimiser import "
            "CallableHost, OptimiserConfig, load_cases, optimise",
            "",
            "cases = load_cases(\"cases.jsonl\")  # one {'id', 'input', 'expected'} per line",
            'host = CallableHost(run_case, surface={"prompt/system": "Be brief."}, cases=cases)',
            "result = await optimise(host, OptimiserConfig(), proposer=model, model=model)",
            "print(result.patch)  # a reviewable diff; nothing is applied",
        ),
        setup=CASES,
    ),
)

VIEWER = Recipe(
    "see runs in a browser (timeline, flow graph, approvals)",
    "`agent_adapter.viewer.record()`, then `agent-adapter ui`",
    (
        "from agent_adapter.viewer import record",
        "",
        "record(executor)  # every run goes to .agent_adapter/runs",
        "# then, in a terminal: agent-adapter ui",
    ),
)

CHAT = (
    Recipe(
        "chat with any agent or model in a local page",
        "`viewer.serve(...)` or `agent-adapter ui --agent`",
        (
            "from agent_adapter.viewer import ChatMessage, ChatSession, serve",
            "",
            "",
            "async def chat(message: ChatMessage, session: ChatSession) -> str:",
            '    return f"you said: {message.text}"  # your agent; tools: ctx=session.context()',
            "",
            "",
            'if __name__ == "__main__":',
            "    serve(agent=chat)  # or serve(model=model, registry=registry)",
            "# CLI: agent-adapter ui --agent app:chat",
        ),
    ),
)
